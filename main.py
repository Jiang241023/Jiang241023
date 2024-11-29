import gin
import logging
import wandb
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import mobilenet_like, vgg_like, inception_v2_like
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', False, 'Specify whether to train or evaluate a model.')

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

def main(argv):

    # generate folder structures
    run_paths_1 = utils_params.gen_run_folder(path_model_id = 'mobilenet_like')
    run_paths_2 = utils_params.gen_run_folder(path_model_id = 'vgg_like')
    run_paths_3 = utils_params.gen_run_folder(path_model_id = 'inception_v2')

    # set loggers
    utils_misc.set_loggers(run_paths_1['path_logs_train'], logging.INFO)
    utils_misc.set_loggers(run_paths_2['path_logs_train'], logging.INFO)
    utils_misc.set_loggers(run_paths_3['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings([r'F:\dl lab\dl-lab-24w-team04-feature\Jiang241023\configs\config.gin'], [])
    utils_params.save_config(run_paths_1['path_gin'], gin.config_str())
    utils_params.save_config(run_paths_2['path_gin'], gin.config_str())
    utils_params.save_config(run_paths_3['path_gin'], gin.config_str())

    # setup wandb
    wandb.init(project='diabetic-retinopathy-detection', name=run_paths_1['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    wandb.init(project='diabetic-retinopathy-detection', name=run_paths_2['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))
    wandb.init(project='diabetic-retinopathy-detection', name=run_paths_3['model_id'],
               config=utils_params.gin_config_to_readable_dictionary(gin.config._CONFIG))

    # setup pipeline
    ds_train, ds_val, ds_test, ds_info, num_batches = datasets.load(name = 'idrid')
    # for images, labels in ds_train.take(1):  # Take 3 batches
    #     print(f"Images: {images.numpy()}")  # Check image tensor values
    #     print(f"Labels: {labels.numpy()}")

     # model
    model_1, base_model_1 = mobilenet_like(input_shape=ds_info["features"]["image"]["shape"],
                                           n_classes=ds_info["features"]["label"]["num_classes"])
    model_2, base_model_2 = vgg_like(input_shape=ds_info["features"]["image"]["shape"],
                                       n_classes=ds_info["features"]["label"]["num_classes"])
    model_3, base_model_3 = inception_v2_like(input_shape=ds_info["features"]["image"]["shape"],
                                       n_classes=ds_info["features"]["label"]["num_classes"])

    if FLAGS.train:

        # Model_1
        train_model(model = model_1,
                    base_model = base_model_1,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    num_batches = num_batches,
                    ds_info = ds_info,
                    run_paths = run_paths_1,
                    path_model_id = 'mobilenet_like')

        # Model_2
        train_model(model = model_2,
                    base_model = base_model_2,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    num_batches = num_batches,
                    ds_info = ds_info,
                    run_paths = run_paths_2,
                    path_model_id = 'vgg_like')

        # Model_3
        train_model(model = model_3,
                    base_model = base_model_3,
                    ds_train = ds_train,
                    ds_val = ds_val,
                    num_batches = num_batches,
                    ds_info = ds_info,
                    run_paths = run_paths_3,
                    path_model_id = 'inception_v2_like')


    else:
        #checkpoint_path_1 = r'F:\dl lab\dl-lab-24w-team04-feature\experiments\run_2024-11-29T18-06-50-456158_mobilenet_like\ckpts'
        #checkpoint_path_2 = r'F:\dl lab\dl-lab-24w-team04-feature\experiments\run_2024-11-29T18-06-50-457157_vgg_like\ckpts'

        checkpoint_path_1 = r'F:\dl lab\dl-lab-24w-team04-feature\experiments\run_2024-11-29T18-59-50-215431_mobilenet_like\ckpts'
        checkpoint_path_2 = r'F:\dl lab\dl-lab-24w-team04-feature\experiments\run_2024-11-29T18-59-50-215431_vgg_like\ckpts'
        checkpoint_path_3 = r'F:\dl lab\dl-lab-24w-team04-feature\experiments\run_2024-11-29T18-59-50-216431_inception_v2\ckpts'

        # Model_1
        checkpoint_1 = tf.train.Checkpoint(model = model_1)
        latest_checkpoint_1 = tf.train.latest_checkpoint(checkpoint_path_1)
        if latest_checkpoint_1:
            print(f"Restoring from checkpoint_1: {latest_checkpoint_1}")
            checkpoint_1.restore(latest_checkpoint_1)
        else:
            print("No checkpoint found. Starting from scratch.")

        # Model_2
        checkpoint_2 = tf.train.Checkpoint(model = model_2)
        latest_checkpoint_2 = tf.train.latest_checkpoint(checkpoint_path_2)
        if latest_checkpoint_2:
            print(f"Restoring from checkpoint_2: {latest_checkpoint_2}")
            checkpoint_2.restore(latest_checkpoint_2)
        else:
            print("No checkpoint found. Starting from scratch.")

        # Model_3
        checkpoint_3 = tf.train.Checkpoint(model = model_3)
        latest_checkpoint_3 = tf.train.latest_checkpoint(checkpoint_path_3)
        if latest_checkpoint_3:
            print(f"Restoring from checkpoint_3: {latest_checkpoint_3}")
            checkpoint_3.restore(latest_checkpoint_3)
        else:
            print("No checkpoint found. Starting from scratch.")

        evaluate(model_1 = model_1, model_2 = model_2, model_3 = model_3, ds_test = ds_test)


if __name__ == "__main__":
    wandb.login(key="40c93726af78ad0b90c6fe3174c18599ecf9f619")
    app.run(main)