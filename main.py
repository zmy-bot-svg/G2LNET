#!/usr/bin/python
# -*- encoding: utf-8 -*-

import datetime
import os
import copy

# Suppress TensorFlow INFO logs (oneDNN message)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import torch.nn as nn
import torchmetrics
from torch_geometric.transforms import Compose

import optuna
import optuna.exceptions
from optuna.integration import TensorBoardCallback
import tensorboard

from model import G2LNet
from utils.dataset_utils import MP18, dataset_split, get_dataloader
from utils.flags import Flags
from utils.train_utils import KerasModel, LRScheduler
from utils.transforms import GetY, GetAngle, ToFloat, AddPeriodicCompleteFeatures

os.environ["NUMEXPR_MAX_THREADS"] = "24"

import logging

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_dataset(config):
    transforms = [GetAngle(), ToFloat()]
    
    use_global_context = getattr(config, 'use_global_context', False)
    if hasattr(config, 'netAttributes'):
        use_global_context = getattr(config.netAttributes, 'use_global_context', use_global_context)
    
    if use_global_context:
        global_radius = getattr(config, 'global_radius', 4.0)
        if hasattr(config, 'netAttributes'):
            global_radius = getattr(config.netAttributes, 'global_radius', global_radius)
        transforms.append(AddPeriodicCompleteFeatures(cutoff=global_radius))
        logging.info("Global features enabled, cutoff radius: %s", global_radius)
    else:
        logging.info("Using pure G2LNet mode")
    
    target_names = getattr(config, 'target_names', ['formation_energy_peratom'])
    if hasattr(config, 'data'):
        target_names = getattr(config.data, 'target_names', target_names)
    
    from utils.transforms import GetMultiTaskY
    if len(target_names) > 1:
        pre_transforms = [GetMultiTaskY()]
        logging.info("Multi-task learning mode: %s targets %s", len(target_names), target_names)
    else:
        pre_transforms = [GetY()]
        logging.info("Single-task learning mode: target %s", target_names[0])
    
    dataset_path = getattr(config, 'dataset_path', './data')
    dataset_name = getattr(config, 'dataset_name', 'jarvis_multitask')
    max_edge_distance = getattr(config, 'max_edge_distance', 8.0)
    n_neighbors = getattr(config, 'n_neighbors', 12)
    edge_input_features = getattr(config, 'edge_input_features', 50)
    points = getattr(config, 'points', 1000)
    
    if hasattr(config, 'data'):
        dataset_path = getattr(config.data, 'dataset_path', dataset_path)
        dataset_name = getattr(config.data, 'dataset_name', dataset_name)
        points = getattr(config.data, 'points', points)
    if hasattr(config, 'netAttributes'):
        max_edge_distance = getattr(config.netAttributes, 'max_edge_distance', max_edge_distance)
        n_neighbors = getattr(config.netAttributes, 'n_neighbors', n_neighbors)
        edge_input_features = getattr(config.netAttributes, 'edge_input_features', edge_input_features)
    
    dataset = MP18(
        root=dataset_path, 
        name=dataset_name, 
        transform=Compose(transforms), 
        pre_transform=pre_transforms,
        r=max_edge_distance, 
        n_neighbors=n_neighbors, 
        edge_steps=edge_input_features, 
        image_selfloop=True, 
        points=points, 
        target_name=target_names
    )
    return dataset

def setup_model(dataset, config):
    def get_param(name, default):
        if hasattr(config, 'netAttributes'):
            return getattr(config.netAttributes, name, getattr(config, name, default))
        return getattr(config, name, default)
    
    target_names = getattr(config, 'target_names', ['formation_energy_peratom'])
    if hasattr(config, 'data'):
        target_names = getattr(config.data, 'target_names', target_names)
    
    use_global_context = get_param('use_global_context', False)
    global_fusion_alpha = get_param('global_fusion_alpha', 0.5)

    atom_input_dim = get_param('atom_input_features', 92)
    try:
        if hasattr(dataset, 'num_features') and dataset.num_features:
            atom_input_dim = int(dataset.num_features)
        else:
            sample = dataset[0]
            if hasattr(sample, 'x') and sample.x is not None:
                atom_input_dim = int(sample.x.shape[1])
    except Exception:
        pass
    
    net = G2LNet(
            data=dataset,
            firstUpdateLayers=get_param('firstUpdateLayers', 4),
            secondUpdateLayers=get_param('secondUpdateLayers', 4),
            atom_input_features=atom_input_dim,
            edge_input_features=get_param('edge_input_features', 50),
            triplet_input_features=get_param('triplet_input_features', 40),
            embedding_features=get_param('embedding_features', 64),
            hidden_features=get_param('hidden_features', 96),
            num_tasks=len(target_names),
            min_edge_distance=get_param('min_edge_distance', 0.0),
            max_edge_distance=get_param('max_edge_distance', 8.0),
            dropout_rate=get_param('dropout_rate', 0.0),
            use_global_context=use_global_context,
            global_fusion_alpha=global_fusion_alpha,
        )
    return net

def setup_optimizer(net, config):
    lr = getattr(config, 'lr', 1e-3)
    optimizer_name = getattr(config, 'optimizer', 'AdamW')
    optimizer_args = getattr(config, 'optimizer_args', {})
    
    if hasattr(config, 'training'):
        lr = getattr(config.training, 'lr', lr)
        optimizer_name = getattr(config.training, 'optimizer', optimizer_name)
        
        if hasattr(config.training, 'optimizer_args'):
            opt_args = config.training.optimizer_args
            if isinstance(opt_args, dict):
                optimizer_args = opt_args
            else:
                optimizer_args = vars(opt_args) if hasattr(opt_args, '__dict__') else {}
    
    if not isinstance(optimizer_args, dict):
        if hasattr(optimizer_args, '__dict__'):
            optimizer_args = vars(optimizer_args)
        else:
            optimizer_args = {}
    
    optimizer = getattr(torch.optim, optimizer_name)(
        net.parameters(),
        lr=lr,
        **optimizer_args
    )
    return optimizer

def setup_schduler(optimizer, config):
    scheduler_name = getattr(config, 'scheduler', 'ReduceLROnPlateau')
    scheduler_args = getattr(config, 'scheduler_args', {})
    
    if hasattr(config, 'training'):
        scheduler_name = getattr(config.training, 'scheduler', scheduler_name)
        if hasattr(config.training, 'scheduler_args'):
            sch_args = config.training.scheduler_args
            if isinstance(sch_args, dict):
                scheduler_args = sch_args
            else:
                scheduler_args = vars(sch_args) if hasattr(sch_args, '__dict__') else {}
    
    if not isinstance(scheduler_args, dict):
        if hasattr(scheduler_args, '__dict__'):
            scheduler_args = vars(scheduler_args)
        else:
            scheduler_args = {}
    
    scheduler = LRScheduler(optimizer, scheduler_name, scheduler_args)
    return scheduler

def build_keras(net, optimizer, scheduler, config):
    from utils.train_utils import MultiTaskL1Loss 
    
    target_names = getattr(config, 'target_names', ['formation_energy_peratom'])
    if hasattr(config, 'data'):
        target_names = getattr(config.data, 'target_names', target_names)
    
    model = KerasModel(
        net=net,
        loss_fn=MultiTaskL1Loss(),
        target_names=target_names,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=config
    )
    return model

def train(config, printnet=False, trial=None):
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    log_enable = getattr(config, 'log_enable', False)
    project_name = getattr(config, 'project_name', 'g2lnet_multitask')
    seed = getattr(config, 'seed', 42)
    batch_size = getattr(config, 'batch_size', 32)
    num_workers = getattr(config, 'num_workers', 0)
    debug = getattr(config, 'debug', False)
    
    if hasattr(config, 'training'):
        log_enable = getattr(config.training, 'log_enable', log_enable)
        project_name = getattr(config.training, 'project_name', project_name)
        seed = getattr(config.training, 'seed', seed)
        batch_size = getattr(config.training, 'batch_size', batch_size)
        num_workers = getattr(config.training, 'num_workers', num_workers)
        debug = getattr(config.training, 'debug', debug)

    dataset = setup_dataset(config)
    train_dataset, val_dataset, test_dataset = dataset_split(
        dataset, train_size=0.8, valid_size=0.1, test_size=0.1, seed=seed, debug=debug) 
    train_loader, val_loader, test_loader = get_dataloader(
        train_dataset, val_dataset, test_dataset, batch_size, num_workers)

    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    callbacks = None
    
    model = build_keras(net, optimizer, scheduler, config)
    
    output_dir = getattr(config, 'output_dir', './output')
    net_name = getattr(config, 'net', 'g2lnet')
    epochs = getattr(config, 'epochs', 100)
    patience = getattr(config, 'patience', 20)
    
    if hasattr(config, 'training'):
        output_dir = getattr(config.training, 'output_dir', output_dir)
        net_name = getattr(config.training, 'net', net_name)
        epochs = getattr(config.training, 'epochs', epochs)
        patience = getattr(config.training, 'patience', patience)
    
    history = model.fit(
        train_loader,
        val_loader,
        ckpt_path=os.path.join(output_dir, net_name+'.pth'),
        epochs=epochs,
        monitor='val_mae',
        mode='min',
        patience=patience,
        plot=True,
        callbacks=callbacks,
        trial=trial
    )
    
    test_result = model.evaluate(test_loader)
    print(f"\nTest results: {test_result}")
    print(f"Model parameter count: {model.total_params():,}")

    best_val_mae = min(history['val_mae']) if 'val_mae' in history else float('inf')
    return best_val_mae

def objective(trial, base_config):
    """
    Optuna optimization objective based on config
    
    Args:
        trial: Optuna trial object
        base_config: base config object
    
    Returns:
        float: objective to minimize (validation MAE)
    """
    config = copy.deepcopy(base_config)
    
    search_space = None
    if hasattr(config, 'hyperparameter_search') and hasattr(config.hyperparameter_search, 'search_space'):
        search_space = config.hyperparameter_search.search_space
        logging.info("Found config-driven search space with %s parameters", len(vars(search_space)))
    elif hasattr(config, 'optuna') and hasattr(config.optuna, 'search_space'):
        search_space = config.optuna.search_space
        logging.info("Found config-driven search space (legacy format) with %s parameters", len(vars(search_space)))
    
    if search_space is not None:
        for param_name, param_config_ns in vars(search_space).items():
            if hasattr(param_config_ns, '__dict__'):
                param_config = vars(param_config_ns)
            else:
                param_config = param_config_ns
            
            param_type = param_config.get('type')
            
            if param_type == 'float':
                value = trial.suggest_float(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                value = trial.suggest_categorical(param_name, param_config['choices'])
            elif param_type == 'int':
                value = trial.suggest_int(
                    param_name, 
                    param_config['low'], 
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")
            
            if param_name == 'lr':
                if hasattr(config, 'training'):
                    config.training.lr = value
                else:
                    config.lr = value
            elif param_name == 'dropout_rate':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.dropout_rate = value
                else:
                    config.dropout_rate = value
            elif param_name == 'weight_decay':
                if hasattr(config, 'training'):
                    if not hasattr(config.training, 'optimizer_args'):
                        from argparse import Namespace
                        config.training.optimizer_args = Namespace()
                    if hasattr(config.training.optimizer_args, '__dict__'):
                        opt_args_dict = vars(config.training.optimizer_args)
                    elif isinstance(config.training.optimizer_args, dict):
                        opt_args_dict = config.training.optimizer_args
                    else:
                        opt_args_dict = {}
                    opt_args_dict['weight_decay'] = value
                    if isinstance(config.training.optimizer_args, dict):
                        config.training.optimizer_args = opt_args_dict
                    else:
                        config.training.optimizer_args = Namespace(**opt_args_dict)
                else:
                    if not hasattr(config, 'optimizer_args'):
                        config.optimizer_args = {}
                    if hasattr(config.optimizer_args, '__dict__'):
                        config.optimizer_args = vars(config.optimizer_args)
                    if not isinstance(config.optimizer_args, dict):
                        config.optimizer_args = {}
                    config.optimizer_args['weight_decay'] = value
            elif param_name == 'batch_size':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.batch_size = value
                else:
                    config.batch_size = value
            elif param_name == 'hidden_features':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.hidden_features = value
                else:
                    config.hidden_features = value
            elif param_name == 'firstUpdateLayers':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.firstUpdateLayers = value
                else:
                    config.firstUpdateLayers = value
            elif param_name == 'secondUpdateLayers':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.secondUpdateLayers = value
                else:
                    config.secondUpdateLayers = value
            elif param_name == 'embedding_features':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.embedding_features = value
                else:
                    config.embedding_features = value
            elif param_name == 'epochs':
                if hasattr(config, 'training'):
                    config.training.epochs = value
                else:
                    config.epochs = value
            elif param_name == 'percnet_cutoff':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.percnet_cutoff = value
                else:
                    config.percnet_cutoff = value
            elif param_name == 'percnet_fusion_alpha':
                if hasattr(config, 'netAttributes'):
                    config.netAttributes.percnet_fusion_alpha = value
                else:
                    config.percnet_fusion_alpha = value
            elif param_name == 'gradnorm_alpha':
                config.gradnorm_alpha = value
            else:
                setattr(config, param_name, value)
    else:
        logging.warning("hyperparameter_search.search_space not found in config, using hard-coded search space")
        
        lr_value = trial.suggest_float("lr", 0.0005, 0.002, log=True)
        if hasattr(config, 'training'):
            config.training.lr = lr_value
        else:
            config.lr = lr_value
        
        dropout_value = trial.suggest_float("dropout_rate", 0.05, 0.2)
        if hasattr(config, 'netAttributes'):
            config.netAttributes.dropout_rate = dropout_value
        else:
            config.dropout_rate = dropout_value
        
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-4, log=True)
        if hasattr(config, 'training') and hasattr(config.training, 'optimizer_args'):
            if hasattr(config.training.optimizer_args, '__dict__'):
                config.training.optimizer_args = vars(config.training.optimizer_args)
            if not isinstance(config.training.optimizer_args, dict):
                config.training.optimizer_args = {}
            config.training.optimizer_args['weight_decay'] = weight_decay
        elif hasattr(config, 'optimizer_args'):
            if hasattr(config.optimizer_args, '__dict__'):
                optimizer_args_dict = vars(config.optimizer_args)
            else:
                optimizer_args_dict = config.optimizer_args if isinstance(config.optimizer_args, dict) else {}
            optimizer_args_dict['weight_decay'] = weight_decay
            config.optimizer_args = optimizer_args_dict
        else:
            config.optimizer_args = {'weight_decay': weight_decay}
        
        firstUpdateLayers_value = trial.suggest_categorical("firstUpdateLayers", [3, 4, 5])
        secondUpdateLayers_value = trial.suggest_categorical("secondUpdateLayers", [3, 4, 5])
        if hasattr(config, 'netAttributes'):
            config.netAttributes.firstUpdateLayers = firstUpdateLayers_value
            config.netAttributes.secondUpdateLayers = secondUpdateLayers_value
        else:
            config.firstUpdateLayers = firstUpdateLayers_value
            config.secondUpdateLayers = secondUpdateLayers_value
        
        hidden_features_value = trial.suggest_categorical("hidden_features", [96, 128, 160, 192])
        if hasattr(config, 'netAttributes'):
            config.netAttributes.hidden_features = hidden_features_value
        else:
            config.hidden_features = hidden_features_value

    print("\n" + "="*60)
    print(f"Starting Trial #{trial.number}")
    print("  Parameters:")
    for key, value in trial.params.items():
        print(f"    - {key}: {value}")
    print("="*60 + "\n")
    
    config.log_enable = False
    
    trial_name = f"trial_{trial.number}"
    config.output_dir = os.path.join(base_config.output_dir, trial_name)
    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        best_val_mae = train(config, printnet=False, trial=trial)
        
        print(f"\nTrial #{trial.number} completed!")
        print(f"   Result: {best_val_mae:.6f}")
        print(f"   Current best: {trial.study.best_value:.6f}" if trial.study.best_value else "   First trial")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_cached = torch.cuda.memory_reserved() / 1e9
            print(f"   GPU Memory: {memory_allocated:.2f}GB allocated, {memory_cached:.2f}GB cached")
        
        print("="*60)
        
        return best_val_mae
    except optuna.exceptions.TrialPruned:
        print(f"\nTrial #{trial.number} pruned!")
        print("   Reason: Performance not promising, stopped early")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("="*60)
        raise
    except torch.cuda.OutOfMemoryError as e:
        print(f"\nTrial #{trial.number} failed: GPU Out of Memory")
        print("   Suggestion: reduce batch_size or hidden_features")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("="*60)
        return 1.0
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("="*60)
        return 1.0

def predict(config):
    dataset = setup_dataset(config)
    from torch_geometric.loader import DataLoader
    test_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=False,)

    rank = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = setup_model(dataset, config).to(rank)

    optimizer = setup_optimizer(net, config)
    scheduler = setup_schduler(optimizer, config)

    model = build_keras(net, optimizer, scheduler, config)
    model.predict(test_loader, ckpt_path=config.model_path, test_out_path=config.output_path)

if __name__ == "__main__":
    # Main entry point
    import warnings
    warnings.filterwarnings('ignore', '.*TypedStorage is deprecated.*')
    
    flags = Flags()
    config = flags.updated_config
    
    name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = os.path.join(config.output_dir, name)
    if not(os.path.exists(config.output_dir)):
        os.makedirs(config.output_dir)
    set_seed(config.seed)

    if config.task_type.lower() == 'train':
        train(config)
    
    elif config.task_type.lower() == 'config_hyperparameter':
        print("Starting config-driven hyperparameter search...")
        
        if hasattr(config, 'hyperparameter_search') and config.hyperparameter_search.get('enabled', False):
            from config import create_objective_from_config
            
            search_config = config.hyperparameter_search
            storage_url = search_config.get('storage_url', f"sqlite:///{config.output_dir}/study.db")
            
            study = optuna.create_study(
                study_name=search_config.get('study_name', config.project_name),
                storage=storage_url,
                direction=search_config.get('direction', 'minimize'),
                sampler=optuna.samplers.TPESampler(seed=config.seed),
                load_if_exists=True
            )
            
            objective = create_objective_from_config(config)
            
            n_trials = search_config.get('n_trials', 50)
            timeout = search_config.get('timeout', None)
            
            print("Search configuration:")
            print(f"   Trials: {n_trials}")
            print(f"   Storage: {storage_url}")
            print(f"   Parameters: {list(search_config['search_space'].keys())}")
            print("="*60)
            
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            print(f"\nSearch finished! Best MAE: {study.best_value:.6f}")
            print(f"Best parameters: {study.best_params}")
            
        else:
            print("Hyperparameter search is not enabled in config. Set hyperparameter_search.enabled = True")
    
    elif config.task_type.lower() == 'hyperparameter':
        print("Starting Optuna hyperparameter optimization...")
        
        db_path = os.path.join(config.output_dir, f"{config.project_name}.db")
        storage_name = f"sqlite:///{db_path}"
        
        tensorboard_dir = os.path.join(config.output_dir, "tensorboard_logs")
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        
        study = optuna.create_study(
            direction="minimize",
            study_name=config.project_name,
            storage=storage_name,
            sampler=optuna.samplers.TPESampler(seed=config.seed),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=1,
                n_warmup_steps=2,
                interval_steps=1
            ),
            load_if_exists=True,
        )
        
        if hasattr(config, 'hyperparameter_search') and hasattr(config.hyperparameter_search, 'search_space'):
            def wrapped_objective(trial):
                return objective(trial, config)
            print("Using config-driven hyperparameter search space")
            
            search_space = config.hyperparameter_search.search_space
            print("Search space from config:")
            for param_name in vars(search_space).keys():
                param_config = getattr(search_space, param_name)
                if hasattr(param_config, '__dict__'):
                    param_dict = vars(param_config)
                else:
                    param_dict = param_config
                
                if param_dict.get('type') == 'categorical':
                    print(f"   - {param_name}: {param_dict['choices']}")
                else:
                    print(f"   - {param_name}: [{param_dict['low']}, {param_dict['high']}] ({param_dict['type']})")
        elif hasattr(config, 'optuna') and hasattr(config.optuna, 'search_space'):
            def wrapped_objective(trial):
                return objective(trial, config)
            print("Using config-driven hyperparameter search space")
            
            search_space = config.optuna.search_space
            print("Search space from config:")
            for param_name in vars(search_space).keys():
                param_config = getattr(search_space, param_name)
                if hasattr(param_config, '__dict__'):
                    param_dict = vars(param_config)
                else:
                    param_dict = param_config
                
                if param_dict.get('type') == 'categorical':
                    print(f"   - {param_name}: {param_dict['choices']}")
                else:
                    print(f"   - {param_name}: [{param_dict['low']}, {param_dict['high']}] ({param_dict['type']})")
        else:
            def wrapped_objective(trial):
                return objective(trial, config)
            print("Using standard hyperparameter search space (fallback mode)")
        
        n_trials = getattr(config.hyperparameter_search, 'n_trials', None) if hasattr(config, 'hyperparameter_search') else None
        if n_trials is None:
            n_trials = getattr(config, 'sweep_count', 50)
        
        print(f"Running {n_trials} trials...")
        print(f"Results will be saved to: {db_path}")
        print(f"TensorBoard logs: {tensorboard_dir}")
        print("To view real-time progress: tensorboard --logdir", tensorboard_dir)
        print("="*60)
        
        study.optimize(
            wrapped_objective,
            n_trials=n_trials,
            callbacks=[TensorBoardCallback(tensorboard_dir, metric_name='val_mae')]
        )
        
        print("="*60)
        print("Hyperparameter optimization finished!")
        
        total_trials = len(study.trials)
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        print("Trial summary:")
        print(f"   Total trials: {total_trials}")
        print(f"   Completed trials: {completed_trials}")
        print(f"   Pruned trials: {pruned_trials} (time saved: {pruned_trials/total_trials*100:.1f}%)")
        print(f"   Failed trials: {failed_trials}")
        
        if study.best_trial:
            print("\nBest result:")
            print(f"   Trial number: #{study.best_trial.number}")
            print(f"   Best validation MAE: {study.best_value:.6f}")
            print(f"   Best parameters:")
            for key, value in study.best_params.items():
                print(f"     {key}: {value}")
        else:
            print("\nNo completed trials")
        
        print("="*60)
        
        results_file = os.path.join(config.output_dir, "optuna_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best value: {study.best_value:.6f}\n")
            f.write("Best parameters:\n")
            for key, value in study.best_params.items():
                f.write(f"  {key}: {value}\n")
        
        print(f"Results saved to: {results_file}")
        print(f"Database saved to: {db_path}")
        print(f"TensorBoard logs saved to: {tensorboard_dir}")
        print(f"To view TensorBoard: tensorboard --logdir {tensorboard_dir}")
        print(f"To generate visualization: python optuna_visualizer.py --study {db_path} --output ./visualization")
    
    
    elif config.task_type.lower() == 'predict':
        predict(config)
    
    else:
        raise NotImplementedError(f"Task type {config.task_type} not implemented. Supported types: train, hyperparameter, config_hyperparameter, cv, predict")