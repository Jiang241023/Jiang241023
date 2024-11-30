import logging
import wandb
import gin
import math

import tensorflow as tf
from input_pipeline.datasets import load
from models.architectures import mobilenet_like, vgg_like, inception_v2_like
from train import Trainer
from utils import utils_params, utils_misc

def train_model(model, base_model, ds_train, ds_val, num_batches, ds_info, run_paths, path_model_id):
    print('-' * 88)
    print(f'Starting training {path_model_id}')
    model.summary()
    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, num_batches)
    for layer in model.layers:
        print(layer.name, layer.trainable)
    for _ in trainer.train():
        continue
    for layer in base_model.layers[-10:]:
        layer.trainable = True
    for _ in trainer.train():
        continue
    print(f"Training checkpoint path for {path_model_id}: {run_paths['path_ckpts_train']}")
    print(f'Training completed for {path_model_id}')
    print('-' * 88)

def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            if isinstance(value, str): # This checks whether the variable value is of type str.
                bindings.append(f"{key}='{value}'")
            else:
                bindings.append(f"{key}={value}")

        # generate folder structures
        model_type = run.config['model_type']
        run_paths = utils_params.gen_run_folder(path_model_id = model_type)


        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)


        # gin-config
        gin.parse_config_files_and_bindings([r'F:\dl lab\dl-lab-24w-team04-feature\Jiang241023\configs\config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())


        # setup pipeline
        #ds_train, ds_val, ds_test, ds_info = load()
        ds_train, ds_val, ds_test, ds_info, num_batches = load(name='idrid')


        # Model
        if model_type == 'mobilenet_like':
            model, base_model = mobilenet_like(input_shape=ds_info["features"]["image"]["shape"],
                                               n_classes=ds_info["features"]["label"]["num_classes"])
        elif model_type == 'vgg_like':
            model, base_model = vgg_like(input_shape=ds_info["features"]["image"]["shape"],
                                         n_classes=ds_info["features"]["label"]["num_classes"])
        elif model_type == 'inception_v2_like':
            model, base_model = inception_v2_like(input_shape=ds_info["features"]["image"]["shape"],
                                                  n_classes=ds_info["features"]["label"]["num_classes"])
        else:
            raise ValueError

        train_model(model = model,
                    base_model = base_model,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    num_batches = num_batches,
                    ds_info = ds_info,
                    run_paths = run_paths,
                    path_model_id = model_type)

model_types = ['mobilenet_like', 'vgg_like', 'inception_v2_like']
for model in model_types:
    sweep_config = {
        'name': 'idrid-sweep',
        'method': 'random',
        'metric': {
            'name': 'val_acc',
            'goal': 'maximize'
        },
        'parameters': {
            'Trainer.total_epochs': {
                'values': [10]
            },
            'model_type':{
                'values': [model]
            },
            'vgg_like.base_filters': {
                'distribution': 'q_log_uniform',
                'q': 1,
                'min': math.log(8), # -> ln8 = 2.0794
                'max': math.log(128) # -> ln128 = 4.852
            },
            'vgg_like.n_blocks': {
                'distribution': 'q_uniform',
                'q': 1,
                'min': 2,
                'max': 6
            },
            'vgg_like.dense_units': {
                'distribution': 'q_log_uniform',
                'q': 1,
                'min': math.log(16),
                'max': math.log(256)
            },
            'vgg_like.dropout_rate': {
                'distribution': 'uniform',
                'min': 0.1,
                'max': 0.9
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config)

    wandb.agent(sweep_id, function=train_func, count=100)