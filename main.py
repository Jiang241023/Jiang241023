import gin
import logging
import wandb
from absl import app, flags

from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.')

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['configs/config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup wandb
    wandb.init(project='diabetic-retinopathy-detection', name=run_paths['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load()


    # model
    model = vgg_like(input_shape=ds_info["features"]["image"]["shape"], n_classes=ds_info["features"]["label"]["num_classes"])

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    else:
        evaluate(model,
                 checkpoint,
                 ds_test,
                 ds_info,
                 run_paths)

if __name__ == "__main__":
    wandb.login(key="40c93726af78ad0b90c6fe3174c18599ecf9f619")
    app.run(main)