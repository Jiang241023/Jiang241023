import gin
import tensorflow as tf

@gin.configurable
def preprocess(image, label, img_height = 256, img_width = 256):
    """Dataset preprocessing: Normalizing and resizing"""

    image = tf.image.resize(image, (img_height, img_width))
    # Normalize image to [0, 1] and resize
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]


    return image, label

augmentation_layer = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.03),  # Approximately ±10 degrees,
        tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),  # small zoom
        tf.keras.layers.RandomBrightness(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),  # Random contrast adjustment up to ±10%
    ])

def augment(image, label):
    """Data augmentation"""
    image = augmentation_layer(image)
    return image, label







