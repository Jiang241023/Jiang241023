from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from input_pipeline import datasets
from models.architectures import mobilenet_like



def grad_cam_visualization(name, batch_size, data_dir, test_data_dir, base_filters, n_blocks, dense_units, dropout_rate, checkpoint_path, img_path):
    # Load ds_train
    ds_train, _, _, ds_info, _ = datasets.load(name=name, batch_size=batch_size, data_dir=data_dir,
                                               test_data_dir=test_data_dir)

    # Load the saved model
    model, base_model = mobilenet_like(input_shape=ds_info["features"]["image"]["shape"],
                                           n_classes=ds_info["features"]["label"]["num_classes"],
                                           base_filters=base_filters,
                                           n_blocks=n_blocks,
                                           dense_units=dense_units,
                                           dropout_rate=dropout_rate
                                           )
    checkpoint_path = checkpoint_path
    checkpoint = tf.train.Checkpoint(model=model)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if latest_checkpoint:
        print(f"Restoring from checkpoint_1: {latest_checkpoint}")
        checkpoint.restore(latest_checkpoint)
    else:
        print("No checkpoint found. Starting from scratch.")

    model_loaded = model

    #print("Checking model layers:")
    for layer in model_loaded.layers:
        # print(f"Layer name: {layer.name}, Layer type: {type(layer)}")
        print(layer.name)

    last_conv_layer = None

    for images, labels in ds_train.take(1):
        dummy_img = images[0:1]
        break
    _ = model_loaded(dummy_img)

    for images, labels in ds_train.take(1):
        dummy_img = images[0:1]
        break
    _ = model_loaded(dummy_img)

    def find_target_layer(model):
        for layer in reversed(model.layers):
            # check to see if the layer has a 4D output
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

    last_conv_layer_name = find_target_layer(model_loaded)

    if last_conv_layer_name:
        print(f"The last conv layer name is {last_conv_layer_name}")
    else:
        print(f"No Convolutional Layers found in the model")

    # Function to generate Grad-CAM
    def grad_cam(model, img_path, ds_train, last_conv_layer_name, target_class_idx=None):

        # Load and preprocess the image
        image = cv2.imread(img_path)  # Load image in BGR format

        if image is None:
            print(f"Error: File not found or cannot be opened at {img_path}")
        else:
            print("Image loaded successfully!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]
        img_array = np.expand_dims(image, axis=0)  # Add batch dimension
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Predict using the model
        predictions = model(img_array, training=False)
        predictions = tf.sigmoid(predictions[:, 0])
        print("Predicted class:", 1 if predictions[0] > 0.5 else 0)

        # Initialize a model for Grad-CAM
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            # Ensure that the input tensor is watched
            tape.watch(img_tensor)

            # Get outputs for the last conv layer and the predictions
            conv_outputs, predictions = grad_model(img_array)

            # Extract the target class score
            class_output = predictions[:, 0]

        # Calculate gradients of the target class score with respect to conv layer output
        grads = tape.gradient(class_output, conv_outputs)
        print(f"grads shape is {grads.shape}")

        # Compute the mean intensity of the gradients for each feature map
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))  # along, B, H, W
        print(f"weight shape is {weights.shape}")

        # Compute the weighted sum of the feature maps
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

        # Apply ReLU to retain only positive values
        cam = tf.maximum(cam, 0)

        # Normalize the heatmap
        cam /= tf.reduce_max(cam) if tf.reduce_max(cam) != 0 else 1

        return cam

    # Function to display Grad-CAM
    def display_grad_cam(heatmap, img, alpha=0.5):

        # Load the image
        image = cv2.imread(img_path)  # Load image in BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Normalize for Matplot lib
        img = np.array(image) / 255.0

        if isinstance(heatmap, tf.Tensor):
            heatmap = heatmap.numpy()

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # resize to. (4,4) to (256,256)

        if heatmap.shape[:2] != img.shape[:2]:
            raise ValueError("The heatmap and image must have the same spatial dimensions.")

        # Overlay heatmap on original image
        colormap = plt.cm.jet(heatmap)[:, :,
                   :3]  # convert the heat map to 3 channel, extract the first 3 channel , ignore the transparancy channel, (256,256,3)

        blended_image = alpha * colormap + (1 - alpha) * img

        # Display images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Blended Image")
        plt.imshow(blended_image)
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    # Load and preprocess a image
    img_path = img_path

    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model_loaded, img_path, ds_train, last_conv_layer_name)

    # Visualize the result
    display_grad_cam(heatmap, img_path)


if __name__ == '__main__':
    name = 'idrid'
    batch_size = 16
    data_dir = r"F:\IDRID_dataset\images_augmented\images_augmented\train"
    test_data_dir = r"F:\IDRID_dataset\images_augmented\images_augmented\test\binary"
    base_filters = 81
    n_blocks = 5
    dense_units = 65
    dropout_rate = 0.1619613221243074
    checkpoint_path = r'F:\dl lab\dl-lab-24w-team04-feature\experiments\run_2024-11-30T18-05-21-229835_mobilenet_like\ckpts'
    img_path = r'F:\IDRID_dataset\images_augmented\images_augmented\train\class_1\IDRiD_001_aug_0.jpg'
    grad_cam_visualization()
