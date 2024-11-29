import logging
import wandb
import gin
import math

import tensorflow as tf
from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc


def train_func():
    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f'{key}={value}')


        # generate folder structures
        run_paths = utils_params.gen_run_folder(path_model_id = 'vgg_like')

        # set loggers
        utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

        # gin-config
        gin.parse_config_files_and_bindings([r'F:\dl lab\dl-lab-24w-team04-feature\Jiang241023\configs\config.gin'], bindings)
        utils_params.save_config(run_paths['path_gin'], gin.config_str())

        # setup pipeline
        #ds_train, ds_val, ds_test, ds_info = load()
        ds_train, ds_val, ds_test, ds_info, num_batches = load(name='idrid')

        # model
        model, base_model = vgg_like(input_shape=ds_info["features"]["image"]["shape"],
                                     n_classes=ds_info["features"]["label"]["num_classes"])
        model.summary()


        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, num_batches)
       # for layer in model.layers:
          #  print(layer.name, layer.trainable)

        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

        for layer in model.layers:
            print(layer.name, layer.trainable)
        for layer in base_model.layers[-10:]:
            layer.trainable = True

        for _ in trainer.train():
            continue
        print(f"Training checkpoint path: {run_paths['path_ckpts_train']}")


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
        'vgg_like.base_filters': {
            'distribution': 'q_log_uniform',
            'q': 1,
            'min': math.log(8),
            'max': math.log(128)
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

wandb.agent(sweep_id, function=train_func, count=50)