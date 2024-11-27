import gin
from tensorflow.data.experimental import AUTOTUNE
import tensorflow as tf
import logging
from preprocessing import preprocess, augment
import tensorflow_datasets as tfds
from tensorflow.keras.utils import image_dataset_from_directory
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


#
data_dir = r"F:\IDRID_dataset\images_augmented\images_augmented\train"
test_data_dir = r"F:\IDRID_dataset\images_augmented\images_augmented\test\binary"
# print(f"Checking data directory: {data_dir}")
# for root, dirs, files in os.walk(data_dir):
#     for file in files:
#         print(f"Found file: {os.path.join(root, file)}")
@gin.configurable
def load(name, batch_size = 16, data_dir = data_dir  , test_data_dir = test_data_dir  , caching=True):
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Load dataset from directory structure, where each subdirectory represents a class,return an object, which is an iterable tuples (image, label)
        full_ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            batch_size=batch_size,
            label_mode='int') # use 'int' for integer label , for classification
        # for image, label in full_ds.take(1):
        #     # 转换为 NumPy 格式
        #     image_np = image.numpy()
        #     # 设置 NumPy 打印选项
        #     np.set_printoptions(precision=3, suppress=True, threshold=np.inf)  # precision 控制小数位数，threshold 控制打印所有值
        #     print(f"Image: {image_np}")
        #     print(f"label : {label}")

        # Calculate the number of examples for shuffle buffer size
        num_examples = sum(1 for _ in full_ds.unbatch())
        for images, _ in full_ds.take(1):
            image_shape = images.shape[1:]
            break

        class_names = full_ds.class_names
        num_classes = len(class_names)

        # Define df_info
        ds_info = {
            "num_examples": num_examples,
            "features": {
                "image": {"shape": image_shape, "dtype": tf.float32},
                "label": {"num_classes": num_classes, "dtype": tf.int64}
            }
        }
        num_batches = int(len(full_ds) * 0.8)
        print(f"no of batches is {num_batches}")

        # Split into training and validation sets
        total_samples = sum(1 for _ in full_ds.unbatch())
        val_size = int(0.2 * total_samples)
        train_size = int(total_samples) - val_size

        # flatten the dataset
        ds_train = full_ds.unbatch().take(train_size)
        ds_val = full_ds.unbatch().skip(train_size)


        # allow the data to be processed in chunks during training and validation
        ds_train = ds_train.batch(batch_size = batch_size)
        ds_val = ds_val.batch(batch_size = batch_size)




        ds_test= None
        if test_data_dir:
            ds_test = tf.keras.preprocessing.image_dataset_from_directory(
                test_data_dir, batch_size = 1, label_mode = 'int'
            )

        # Prepare and return the training and validation datasets
        return prepare(ds_train, ds_val, num_batches, ds_test=ds_test, ds_info=ds_info, caching=caching)


    elif name == "eyepacs":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'diabetic_retinopathy_detection/btgraham-300',
            split=['train', 'validation', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        def _preprocess(img_label_dict):

            return img_label_dict['image'], img_label_dict['label']

        ds_train = ds_train.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_val = ds_val.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.map(_preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logging.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            'mnist',
            split=['train[:90%]', 'train[90%:]', 'test'],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, num_batches, ds_test = None, ds_info=None, caching=True):

    """Prepare datasets with preprocessing, augmentation, batching, caching, and prefetching"""
    # Prepare training dataset
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
   # for images, labels in ds_train.take(1):
    #    print(f"After augment - Images Min: {tf.reduce_min(images).numpy()}, Max: {tf.reduce_max(images).numpy()}")
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
  #  for images, labels in ds_train.take(1):
  #      print(
 #           f"After preprocess - Images Min: {tf.reduce_min(images).numpy()}, Max: {tf.reduce_max(images).numpy()}")

    if ds_info:
       shuffle_buffer_size = ds_info.get("num_examples", 1000) // 10  # Default to 1000 if ds_info not provided
       ds_train = ds_train.shuffle(shuffle_buffer_size)
    else:
        ds_train = ds_train.shuffle(1000)  # Fallback shuffle size

    ds_train = ds_train.repeat().prefetch(tf.data.AUTOTUNE)

    # Prepare validation dataset (no augmentation)
    ds_val = ds_val.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

    # Prepare test dataset if available (no augmentation)
    if ds_test is not None:
        ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        if caching:
            ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


    return ds_train, ds_val, ds_test, ds_info , num_batches



# ds_train, ds_val, ds_test, ds_info = load(name = 'idrid')
# for images, labels in ds_train.take(4):
#     # Denormalize if images are normalized (for example, using mean and std)
#     def denormalize(image):
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         return image * std + mean
#
#
#     denormalized_image = denormalize(images[0])
#     denormalized_image = tf.clip_by_value(denormalized_image, 0, 1)  # Clip to valid range
#     num_images = 5
#     # Show the image
#     plt.figure(figsize=(15,5))
#     for i in range(num_images):
#         plt.subplot(1 ,num_images,i + 1)
#         plt.imshow(denormalized_image[i].numpy())
#         plt.title(f"label : {labels[i]}")
#         plt.axis("off")
#     plt.show()
#     break
