import tensorflow as tf
import logging
import keras
from keras import layers

batch_size = 16

data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(factor=0.03),  # Approximately ±10 degrees,
    keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # small zoom
    keras.layers.RandomBrightness(0.1),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomFlip("horizontal_and_vertical")
])


def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label


def preprocess(image, label, img_height=256, img_width=256):
    """Dataset preprocessing: Normalizing and resizing"""
    image = tf.image.resize(image, (img_height, img_width))
    # Normalize image to [0, 1] and resize
    image = tf.cast(image, tf.float32) / 255.0

    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

    return image, label


def load(name, data_dir, batch_size=batch_size, caching=True):
    """Load datasets based on name"""
    if name == "idrid":
        logging.info(f"Preparing dataset {name}...")

        # Load dataset from directory structure, where each subdirectory represents a class,return an object, which is an iterable tuples (image, label)
        full_ds = keras.preprocessing.image_dataset_from_directory(
            data_dir,
            batch_size=batch_size,
            label_mode='int'  # use 'int' for integer label , for classification
        )

        # Calculate the number of examples for shuffle buffer size
        num_examples = sum(1 for _ in full_ds.unbatch()) ##########
        ds_info = {"num_examples ": num_examples}

        # Split into training and validation sets
        total_samples = sum(1 for _ in full_ds.unbatch())
        val_size = int(0.2 * total_samples)
        train_size = int(total_samples) - val_size
        ds_train = full_ds.take(train_size)
        ds_val = full_ds.skip(train_size)

        # Prepare and return the training and validation datasets
        return prepare(ds_train, ds_val, ds_info=ds_info, batch_size=batch_size, caching=caching)


def prepare(ds_train, ds_val, ds_test=None, ds_info=None, batch_size = batch_size, caching=True):
    """Prepare datasets with preprocessing, augmentation, batching, caching, and prefetching"""
    # Prepare training dataset
    ds_train = ds_train.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if caching:
        ds_train = ds_train.cache()
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

    return ds_train, ds_val, ds_test, ds_info


data_dir = r'F:\IDRID_dataset\images_augmented\train'

ds_train, ds_val, _, _ = load("idrid", data_dir)

# for images, labels in ds_train.take(1):
#   print("image batch shape: ", images.shape)
#  print("Label batch shape :", labels.shape)




# Load the pretrained Model VGG16
base_model = keras.applications.VGG16(
    input_shape=(256, 256, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the base model
base_model.trainable = False

# Build the model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024),  # Dense layer without activation
    layers.LeakyReLU(negative_slope = 0.01),  # LeakyReLU with a small negative slope
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

small_ds = ds_train.take(10)

# Train the model

history = model.fit(small_ds, validation_data=ds_val, epochs=40, verbose=1)


