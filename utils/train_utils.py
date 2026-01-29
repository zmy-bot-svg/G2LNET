import torch
import torch.nn as nn
import torchmetrics
import sys,datetime
import logging

from tqdm import tqdm 

from copy import deepcopy

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', 200)

import torch

from accelerate import Accelerator


class MultiTaskL1Loss(nn.Module):
    """Multi-task L1 loss that ignores NaNs."""
    def __init__(self):
        super(MultiTaskL1Loss, self).__init__()
        self.l1_loss = nn.L1Loss(reduction='sum')

    def forward(self, preds, labels, weights):
        if preds.dim() == 2 and labels.dim() == 1:
            batch_size, num_tasks = preds.shape
            if labels.numel() == batch_size * num_tasks:
                labels = labels.view(batch_size, num_tasks)
        
        if preds.dim() == 1:
            preds = preds.unsqueeze(1)
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        task_losses_dict = {}
        weighted_losses = []
        unweighted_losses_tensor = []

        for i in range(labels.shape[1]):
            task_preds = preds[:, i]
            task_labels = labels[:, i]

            valid_mask = ~torch.isnan(task_labels)

            if valid_mask.sum() > 0:
                masked_preds = task_preds[valid_mask]
                masked_labels = task_labels[valid_mask]
                
                unweighted_loss = self.l1_loss(masked_preds, masked_labels) / valid_mask.sum()
                task_losses_dict[f"loss_task_{i}"] = unweighted_loss.item()
                weighted_losses.append(weights[i] * unweighted_loss)
                unweighted_losses_tensor.append(unweighted_loss)
            else:
                task_losses_dict[f"loss_task_{i}"] = 0.0
                unweighted_losses_tensor.append(torch.tensor(0.0, device=preds.device))

        if not weighted_losses:
            return (torch.tensor(0.0, device=preds.device, requires_grad=True), 
                    task_losses_dict, 
                    torch.stack(unweighted_losses_tensor))

        total_weighted_loss = torch.stack(weighted_losses).sum()
        
        return total_weighted_loss, task_losses_dict, torch.stack(unweighted_losses_tensor)

def colorful(obj,color="red", display_type="plain"):
    color_dict = {"black":"30", "red":"31", "green":"32", "yellow":"33",
                    "blue":"34", "purple":"35","cyan":"36",  "white":"37"}
    display_type_dict = {"plain":"0","highlight":"1","underline":"4",
                "shine":"5","inverse":"7","invisible":"8"}
    s = str(obj)
    color_code = color_dict.get(color,"")
    display  = display_type_dict.get(display_type,"")
    out = '\033[{};{}m'.format(display,color_code)+s+'\033[0m'
    return out 


class StepRunner:
    def __init__(self, net, loss_fn, accelerator, stage = "train", metrics_dict = None,
                 optimizer = None, lr_scheduler = None
                 ):
        self.net,self.loss_fn,self.metrics_dict,self.stage = net,loss_fn,metrics_dict,stage
        self.optimizer,self.lr_scheduler = optimizer,lr_scheduler
        self.accelerator = accelerator
        self.target_names = list(self.metrics_dict.keys()) if self.metrics_dict else []

    def __call__(self, batch):
        features,labels = batch,batch.y
        
        preds = self.net(features)
        
        if preds.dim() != labels.dim():
            if preds.dim() == 1 and labels.dim() == 2:
                labels = labels.squeeze(-1)
            elif preds.dim() == 2 and labels.dim() == 1:
                preds = preds.squeeze(-1)

        if self.stage == "train":
            _, task_losses_dict, per_task_losses = self.loss_fn(preds, labels, self.net.loss_weights)

            if not hasattr(self.net, 'initial_losses'):
                self.net.initial_losses = per_task_losses.detach().clone()

            weighted_total_loss = torch.sum(per_task_losses * self.net.loss_weights)

            self.optimizer.zero_grad()
            self.accelerator.backward(weighted_total_loss, retain_graph=True)

            last_shared_layer_params = self.net.get_last_shared_layer().parameters()

            avg_grad_norm = []
            for i in range(len(self.target_names)):
                g_i = torch.autograd.grad(
                    per_task_losses[i],
                    list(last_shared_layer_params),
                    retain_graph=True,
                    allow_unused=True
                )
                g_i_list = [g.view(-1) for g in g_i if g is not None]
                if len(g_i_list) == 0:
                    g_i_norm = torch.tensor(0.0, device=per_task_losses.device)
                else:
                    g_i_vec = torch.cat(g_i_list)
                    g_i_norm = torch.norm(g_i_vec)
                avg_grad_norm.append(self.net.loss_weights[i] * g_i_norm.detach())

            avg_grad_norm = torch.stack(avg_grad_norm)
            mean_norm = avg_grad_norm.mean().detach()

            loss_ratio = per_task_losses.detach() / (self.net.initial_losses + 1e-8)
            relative_rates = loss_ratio / (loss_ratio.mean() + 1e-8)
            target_norm = mean_norm * (relative_rates ** self.net.gradnorm_alpha)

            gradnorm_loss = torch.abs(avg_grad_norm - target_norm).sum()

            if self.net.loss_weights.grad is not None:
                self.net.loss_weights.grad.zero_()

            gw_grads = torch.autograd.grad(gradnorm_loss, self.net.loss_weights, allow_unused=False)[0]
            self.net.loss_weights.grad = gw_grads

            self.optimizer.step()

            with torch.no_grad():
                normalize_coeff = len(self.target_names) / self.net.loss_weights.sum()
                self.net.loss_weights.data.mul_(normalize_coeff)

            loss = weighted_total_loss.detach()

        else:
            loss, task_losses_dict, _ = self.loss_fn(preds, labels, torch.ones_like(self.net.loss_weights))
            loss = loss / len(self.target_names)

        all_preds = self.accelerator.gather(preds)
        all_labels = self.accelerator.gather(labels)
        all_loss = self.accelerator.gather(loss).mean()

        if all_preds.dim() == 2 and all_labels.dim() == 1:
            batch_size, num_tasks = all_preds.shape
            if all_labels.numel() == batch_size * num_tasks:
                all_labels = all_labels.view(batch_size, num_tasks)

        step_losses = {self.stage+"_loss":all_loss.item()}

        for i, name in enumerate(self.target_names):
            loss_key = f"{self.stage}_loss_{name}"
            step_losses[loss_key] = task_losses_dict.get(f"loss_task_{i}", 0.0)

        step_metrics = {}
        if self.metrics_dict:
            for i, name in enumerate(self.target_names):
                task_preds = all_preds[:, i]
                task_labels = all_labels[:, i]
                
                valid_mask = ~torch.isnan(task_labels)
                if valid_mask.sum() > 0:
                    metric_fn = self.metrics_dict[name]
                    metric_fn.update(task_preds[valid_mask], task_labels[valid_mask])

        if self.stage=="train":
            if self.optimizer is not None:
                step_metrics['lr'] = self.optimizer.state_dict()['param_groups'][0]['lr']
            else:
                step_metrics['lr'] = 0.0
        return step_losses,step_metrics


class EpochRunner:
    def __init__(self,steprunner,quiet=False,epoch=0):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage=="train" else self.steprunner.net.eval()
        self.accelerator = self.steprunner.accelerator
        self.quiet = quiet
        self.epoch = epoch

    def __call__(self,dataloader):
        loop = tqdm(enumerate(dataloader,start=1),
                    total =len(dataloader),
                    file=sys.stdout,
                    disable=not self.accelerator.is_local_main_process or self.quiet,
                    ncols = 150
                   )
        epoch_losses = {}
        epoch_log = {}
        
        for step, batch in loop:
            if self.stage=="train":
                step_losses,step_metrics = self.steprunner(batch)
            else:
                with torch.no_grad():
                    step_losses,step_metrics = self.steprunner(batch)

            for k,v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k,0.0)+v

            postfix_log = {}
            total_loss_key = f"{self.stage}_loss"
            if total_loss_key in epoch_losses:
                postfix_log[total_loss_key] = f"{(epoch_losses[total_loss_key] / step):.4f}"
            
            if self.stage == "train" and "lr" in step_metrics:
                postfix_log["lr"] = f"{step_metrics['lr']:.2e}"

            loop.set_postfix(**postfix_log)

        epoch_losses = {k: v / len(dataloader) for k, v in epoch_losses.items()}
        epoch_metrics = {}
        if self.steprunner.metrics_dict:
            for name, metric_fn in self.steprunner.metrics_dict.items():
                metric_key = self.stage + "_mae_" + name 
                try:
                    epoch_metrics[metric_key] = metric_fn.compute().item()
                except RuntimeError as e:
                    logging.warning("Could not compute metric for %s: %s", name, e)
                    epoch_metrics[metric_key] = -1.0
                
                metric_fn.reset()
        epoch_log = dict(epoch_losses, **epoch_metrics)

        if self.stage == "val":
            all_task_maes = [v for k, v in epoch_log.items() if 'mae_' in k and k.startswith('val_')]
            if all_task_maes:
                epoch_log['val_mae'] = np.mean(all_task_maes)
            else:
                epoch_log['val_mae'] = epoch_log.get('val_loss', float('inf'))

        if self.accelerator.is_local_main_process and not self.quiet and self.stage == "val":
            log_str = f"Epoch {self.epoch} Summary: "
            
            log_items = []
            if 'val_loss' in epoch_log:
                log_items.append(f"val_loss: {epoch_log['val_loss']:.4f}")
            if 'val_mae' in epoch_log:
                log_items.append(f"val_mae: {epoch_log['val_mae']:.4f}")
            
            for name in self.steprunner.target_names:
                loss_key = f"val_loss_{name}"
                mae_key = f"val_mae_{name}"
                if loss_key in epoch_log and mae_key in epoch_log:
                    log_items.append(f"| {name} -> loss: {epoch_log[loss_key]:.4f}, mae: {epoch_log[mae_key]:.4f}")

            self.accelerator.print(log_str + " ".join(log_items))
            
            val_dict = {k: v for k, v in epoch_log.items() if k.startswith('val_')}
            self.accelerator.print(val_dict)

        return epoch_log


class KerasModel(torch.nn.Module):
    
    StepRunner,EpochRunner = StepRunner,EpochRunner
    
    def __init__(self, net, loss_fn, target_names=None, optimizer=None, lr_scheduler=None, config=None):
        super().__init__()
        self.net, self.loss_fn = net, loss_fn
        self.target_names = target_names if target_names else ['default_task']

        self.net.loss_weights = torch.nn.Parameter(torch.ones(len(self.target_names), dtype=torch.float32))

        self.gradnorm_alpha = config.gradnorm_alpha if hasattr(config, 'gradnorm_alpha') else 1.5
        self.net.gradnorm_alpha = self.gradnorm_alpha

        metrics_dict = torch.nn.ModuleDict({
            name: torchmetrics.MeanAbsoluteError() for name in self.target_names
        })
        self.metrics_dict = metrics_dict

        all_params = list(self.net.parameters())
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            all_params, lr=1e-3)
        self.lr_scheduler = lr_scheduler
        self.from_scratch = True

    def load_ckpt(self, ckpt_path='checkpoint.pt'):
        self.net= torch.load(ckpt_path, weights_only=False)
        self.from_scratch = False

    def forward(self, x):
        return self.net.forward(x)
    
    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min",
            mixed_precision='no', callbacks=None, plot=True, quiet=True, trial=None):
        """Train the model; supports Optuna pruning."""
        self.__dict__.update(locals())

        self.accelerator = Accelerator(mixed_precision=mixed_precision)

        if self.accelerator.is_local_main_process:
            pass
        
        device = str(self.accelerator.device)
        device_type = 'cpu' if 'cpu' in device else 'gpu'
        self.accelerator.print(
            colorful("<<<<<< " + device_type + " " + device + " is used >>>>>>"))

        self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict, self.optimizer, self.lr_scheduler)
        
        train_dataloader, val_dataloader = self.accelerator.prepare(train_data, val_data)
        
        self.history = {}
        callbacks = callbacks if callbacks is not None else []
        
        if plot == True:
            from utils.keras_callbacks import VisProgress
            callbacks.append(VisProgress(self))

        self.callbacks = self.accelerator.prepare(callbacks)
        
        if self.accelerator.is_local_main_process:
            for callback_obj in self.callbacks:
                callback_obj.on_fit_start(model=self)
        
        start_epoch = 1 if self.from_scratch else 0
        
        for epoch in range(start_epoch, epochs + 1):
            import time
            start_time = time.time()
            
            should_quiet = False if quiet == False else (quiet == True or epoch > quiet)
            
            if not should_quiet and self.accelerator.is_local_main_process:
                nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logging.info("%s | Epoch %s / %s", nowtime, epoch, epochs)

            train_step_runner = self.StepRunner(
                net=self.net,
                loss_fn=self.loss_fn,
                accelerator=self.accelerator,
                stage="train",
                metrics_dict=deepcopy(self.metrics_dict),
                optimizer=self.optimizer if epoch > 0 else None,
                lr_scheduler=self.lr_scheduler if epoch > 0 else None
            )

            train_epoch_runner = self.EpochRunner(train_step_runner, should_quiet, epoch=epoch)
        
            train_metrics = {'epoch': epoch}
            train_metrics.update(train_epoch_runner(train_dataloader))

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_train_epoch_end(model=self)

            if val_dataloader:
                val_step_runner = self.StepRunner(
                    net=self.net,
                    loss_fn=self.loss_fn,
                    accelerator=self.accelerator,
                    stage="val",
                    metrics_dict=deepcopy(self.metrics_dict)
                )
                val_epoch_runner = self.EpochRunner(val_step_runner, should_quiet, epoch=epoch)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_dataloader)
                    
                    if 'val_mae' not in val_metrics:
                        val_metrics['val_mae'] = val_metrics.get('val_loss', float('inf'))
                    
                    if self.lr_scheduler.scheduler_type == "ReduceLROnPlateau":
                        monitor_value = val_metrics.get(monitor, 0.0) 
                        self.lr_scheduler.step(metrics=monitor_value)
                    else:
                        self.lr_scheduler.step()
                        
                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]

            if trial is not None and val_dataloader:
                val_mae = val_metrics.get('val_mae')
                if val_mae is None:
                    for key, value in val_metrics.items():
                        if 'mae_' in key and key.startswith('val_'):
                            val_mae = value
                            break
                    if val_mae is None:
                        val_mae = val_metrics.get(monitor, float('inf'))
                
                trial.report(val_mae, epoch)
                
                if trial.should_prune():
                    if not should_quiet:
                        self.accelerator.print(colorful(
                            f"Trial被剪枝在epoch {epoch}, {monitor}: {val_mae:.6f}"))
                    import optuna
                    raise optuna.exceptions.TrialPruned()

            self.accelerator.wait_for_everyone()
            arr_scores = self.history[monitor]
            self.history['best_val_mae'] = self.history.get('best_val_mae', []) + [
                np.min(arr_scores) if mode == "min" else np.max(arr_scores)]

            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)

            if best_score_idx == len(arr_scores) - 1:
                self.accelerator.save(self.net, ckpt_path)
                if not should_quiet:
                    self.accelerator.print(colorful("<<<<<< reach best {0} : {1} >>>>>>".format(
                        monitor, arr_scores[best_score_idx])))

            end_time = time.time()
            self.history['time'] = self.history.get('time', []) + [end_time - start_time]
        
            if len(arr_scores) - best_score_idx > patience:
                self.accelerator.print(colorful(
                    "<<<<<< {} without improvement in {} epoch,""early stopping >>>>>>"
                ).format(monitor, patience))
                break
            
            if self.accelerator.is_local_main_process:
                for callback_obj in self.callbacks:
                    callback_obj.on_validation_epoch_end(model=self)
        
        if self.accelerator.is_local_main_process:
            dfhistory = pd.DataFrame(self.history)
            # self.accelerator.print(dfhistory)
            
            for callback_obj in self.callbacks:
                callback_obj.on_fit_end(model=self)
        
            self.net = self.accelerator.unwrap_model(self.net)
            self.net = torch.load(ckpt_path, weights_only=False)
            return dfhistory

    @torch.no_grad()
    def evaluate(self, val_data):
        accelerator = Accelerator()
        self.net, self.loss_fn, self.metrics_dict = accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict)
        val_data = accelerator.prepare(val_data)
        
        val_step_runner = self.StepRunner(
            net=self.net, stage="val",
            loss_fn=self.loss_fn, 
            metrics_dict=deepcopy(self.metrics_dict),
            accelerator=accelerator
        )
        
        val_epoch_runner = self.EpochRunner(val_step_runner, epoch=0)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics
        
    @torch.no_grad()
    def predict(self, test_data, ckpt_path, test_out_path='test_out.csv'):
        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        
        targets = []
        outputs = []
        id = []
        
        for data in test_data:
            data = data.to(torch.device('cuda'))
            targets.append(data.y.cpu().numpy().tolist())
            output = self.net(data)
            outputs.append(output.cpu().numpy().tolist())
            id += data.structure_id
        
        targets = sum(targets, [])
        outputs = sum(outputs, [])
        id = sum(sum(id, []), [])
        
        import csv
        rows = zip(id, targets, outputs)
        with open(test_out_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for row in rows:
                writer.writerow(row)

    @torch.no_grad()
    def cubic(self, test_data, ckpt_path, test_out_path='cubic_out.csv'):
        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        
        targets = []
        outputs = []
        id = []
        
        for data in test_data:
            data = data.to(torch.device('cuda'))
            targets.append(data.y.cpu().numpy().tolist())
            output = self.net(data)
            outputs.append(output.cpu().numpy().tolist())
            id += data.structure_id
        
        targets = sum(targets, [])
        outputs = sum(outputs, [])
        id = sum(sum(id, []), [])
        
        import csv
        rows = zip(id, targets, outputs)
        with open(test_out_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            for row in rows:
                writer.writerow(row)

    @torch.no_grad()
    def analysis(self, net_name, test_data, ckpt_path, tsne_args, tsne_file_path="tsne_output.png"):
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        
        inputs = []
        def hook(module, input, output):
            inputs.append(input)

        self.ckpt_path = ckpt_path
        self.load_ckpt(self.ckpt_path)
        self.net.eval()
        
        if net_name in ["ALIGNN", "CLIGNN", "GCPNet"]:
            self.net.fc.register_forward_hook(hook)
        else:
            self.net.post_lin_list[0].register_forward_hook(hook)

        targets = []
        for data in test_data:
            data = data.to(torch.device('cuda'))
            targets.append(data.y.cpu().numpy().tolist())
            _ = self.net(data)

        targets = sum(targets, [])
        inputs = [i for sub in inputs for i in sub]
        inputs = torch.cat(inputs)
        inputs = inputs.cpu().numpy()
        
        print("Number of samples: ", inputs.shape[0])
        print("Number of features: ", inputs.shape[1])

        tsne = TSNE(**tsne_args)
        tsne_out = tsne.fit_transform(inputs)

        fig, ax = plt.subplots()
        main = plt.scatter(tsne_out[:, 1], tsne_out[:, 0], c=targets, s=3, cmap='coolwarm')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = plt.colorbar(main, ax=ax)
        import numpy as np
        stdev = np.std(targets)
        cbar.mappable.set_clim(
            np.mean(targets) - 2 * np.std(targets), 
            np.mean(targets) + 2 * np.std(targets)
        )
        plt.savefig(tsne_file_path, format="png", dpi=600)
        plt.show()

    def total_params(self):
        return self.net.total_params()


class LRScheduler:
    """Learning rate scheduler wrapper."""
    def __init__(self, optimizer, scheduler_type, model_parameters):
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type

        self.scheduler = getattr(torch.optim.lr_scheduler, self.scheduler_type)(
            optimizer, **model_parameters
        )

        self.lr = self.optimizer.param_groups[0]["lr"]

    @classmethod
    def from_config(cls, optimizer, optim_config):
        scheduler_type = optim_config["scheduler_type"]
        scheduler_args = optim_config["scheduler_args"]
        return cls(optimizer, scheduler_type, **scheduler_args)

    def step(self, metrics=None, epoch=None):
        if self.scheduler_type == "Null":
            return
        if self.scheduler_type == "ReduceLROnPlateau":
            if metrics is None:
                raise Exception("Validation set required for ReduceLROnPlateau.")
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()

        self.update_lr()

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            self.lr = param_group["lr"]