import logging
import gin

import ray
from ray import tune

from input_pipeline.datasets import load
from models.architectures import vgg_like
from train import Trainer
from utils import utils_params, utils_misc


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id = 'vgg_like')

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings([r'F:\dl lab\dl-lab-24w-team04-feature\Jiang241023\configs\config.gin'], bindings) # change path to absolute path of config file
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    ds_train, ds_val, ds_test, ds_info, num_batches = load(name='idrid')

    # model
    model, base_model = vgg_like(input_shape=ds_info["features"]["image"]["shape"],
                                 n_classes=ds_info["features"]["label"]["num_classes"])
    model.summary()

    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, num_batches)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)


ray.init(num_cpus=10, num_gpus=1)
analysis = tune.run(
    train_func, num_samples=2, resources_per_trial={"cpu": 2, "gpu": 0},
    config={
        "Trainer.total_epochs": tune.grid_search([10]),
        "vgg_like.base_filters": tune.choice([8, 16]),
        "vgg_like.n_blocks": tune.choice([2, 3, 4, 5]),
        "vgg_like.dense_units": tune.choice([32, 64]),
        "vgg_like.dropout_rate": tune.uniform(0, 0.9),
    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()

