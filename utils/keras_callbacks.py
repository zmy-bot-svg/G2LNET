import os
import sys
import datetime
from copy import deepcopy
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from argparse import Namespace

from utils.train_utils import KerasModel


def plot_metric(dfhistory, metric):
    import plotly.graph_objs as go
    train_metrics = dfhistory["train_"+metric].values.tolist()
    val_metrics = dfhistory['val_'+metric].values.tolist()
    epochs = list(range(1, len(train_metrics) + 1))

    train_scatter = go.Scatter(x=epochs, y=train_metrics, mode="lines+markers",
                               name='train_'+metric, marker=dict(size=8, color="blue"),
                               line=dict(width=2, color="blue", dash="dash"))
    val_scatter = go.Scatter(x=epochs, y=val_metrics, mode="lines+markers",
                             name='val_'+metric, marker=dict(size=10, color="red"),
                             line=dict(width=2, color="red", dash="solid"))
    fig = go.Figure(data=[train_scatter, val_scatter])
    return fig


class TensorBoardCallback:
    def __init__(self, save_dir="runs", model_name="model",
                 log_weight=False, log_weight_freq=5):
        self.__dict__.update(locals())
        nowtime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(save_dir, model_name, nowtime)
        self.writer = SummaryWriter(self.log_path)

    def on_fit_start(self, model: 'KerasModel'):
        if self.log_weight:
            net = model.accelerator.unwrap_model(model.net)
            for name, param in net.named_parameters():
                self.writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), 0)
            self.writer.flush()

    def on_train_epoch_end(self, model: 'KerasModel'):

        epoch = max(model.history['epoch'])
        net = model.accelerator.unwrap_model(model.net)
        if self.log_weight and epoch % self.log_weight_freq == 0:
            for name, param in net.named_parameters():
                self.writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()

    def on_validation_epoch_end(self, model: "KerasModel"):

        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        epoch = max(model.history['epoch'])

        dic = deepcopy(dfhistory.iloc[n-1])
        dic.pop("epoch")

        metrics_group = {}
        for key, value in dic.items():
            g = key.replace("train_", '').replace("val_", '')
            metrics_group[g] = dict(metrics_group.get(g, {}), **{key: value})
        for group, metrics in metrics_group.items():
            self.writer.add_scalars(group, metrics, epoch)
        self.writer.flush()

    def on_fit_end(self, model: "KerasModel"):
        epoch = max(model.history['epoch'])
        if self.log_weight:
            net = model.accelerator.unwrap_model(model.net)
            for name, param in net.named_parameters():
                self.writer.add_histogram(
                    name, param.clone().cpu().data.numpy(), epoch)
            self.writer.flush()
        self.writer.close()

        # save history
        dfhistory = pd.DataFrame(model.history)
        dfhistory.to_csv(os.path.join(
            self.log_path, 'dfhistory.csv'), index=None)


class MiniLogCallback:
    def __init__(self, ):
        pass

    def on_fit_start(self, model: 'KerasModel'):
        pass

    def on_train_epoch_end(self, model: 'KerasModel'):
        pass

    def on_validation_epoch_end(self, model: "KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        epoch = max(dfhistory['epoch'])
        monitor_arr = dfhistory[model.monitor]
        best_monitor_score = monitor_arr.max() if model.mode == 'max' else monitor_arr.min()

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"epoch【{epoch}】@{nowtime} --> best_{model.monitor} = {str(best_monitor_score)}",
              file=sys.stderr, end='\r')

    def on_fit_end(self, model: "KerasModel"):
        pass


class VisProgress:
    def __init__(self, figsize=(6, 4)):
        self.figsize = (6, 4)

    def on_fit_start(self, model: 'KerasModel'):
        from .fastprogress import master_bar
        self.mb = master_bar(range(model.epochs))
        self.metric = model.monitor.replace('val_', '')
        dfhistory = pd.DataFrame(model.history)
        x_bounds = [0, min(10, model.epochs)]
        title = f'best {model.monitor} = ?'
        self.mb.update_graph(
            dfhistory, self.metric, x_bounds=x_bounds, title=title, figsize=self.figsize)
        self.mb.update(0)
        self.mb.show()

    def on_train_epoch_end(self, model: 'KerasModel'):
        pass

    def get_title(self, model: 'KerasModel'):
        dfhistory = pd.DataFrame(model.history)
        arr_scores = dfhistory[model.monitor]
        best_score = np.max(
            arr_scores) if model.mode == "max" else np.min(arr_scores)

        title = f'best {model.monitor} = {best_score:.4f}'
        return title

    def on_validation_epoch_end(self, model: "KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        n = len(dfhistory)
        x_bounds = [dfhistory['epoch'].min(), min(10+(n//10)*10, model.epochs)]
        title = self.get_title(model)
        self.mb.update_graph(dfhistory, self.metric, x_bounds=x_bounds,
                             title=title, figsize=self.figsize)
        self.mb.update(n)
        if n == 1:
            self.mb.write(dfhistory.columns, table=True)  
        self.mb.write(dfhistory.iloc[n-1], table=True) 

    def on_fit_end(self, model: "KerasModel"):
        dfhistory = pd.DataFrame(model.history)
        title = self.get_title(model)
        self.mb.update_graph(dfhistory, self.metric,
                             title=title, figsize=self.figsize)
        self.mb.on_iter_end()
