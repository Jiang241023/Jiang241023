# GRAD CAM visualization
import gin
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import os
from input_pipeline import datasets
from models.architectures import mobilenet_like
from train import Trainer

@gin.configurable
def deep_visualization(data_dir, test_data_dir):

    # Load ds_train
    ds_train, _, _, ds_info, _ = datasets.load(name = 'idrid', data_dir = data_dir, test_data_dir = test_data_dir)

    # Load the saved model
    model_1, base_model_1 = mobilenet_like(input_shape = ds_info["features"]["image"]["shape"],
                                           n_classes = ds_info["features"]["label"]["num_classes"])
    checkpoint_path_1 = r'F:\dl lab\dl-lab-24w-team04-feature\experiments\run_2024-11-30T18-05-21-229835_mobilenet_like\ckpts'
    checkpoint_1 = tf.train.Checkpoint(model = model_1)
    latest_checkpoint_1 = tf.train.latest_checkpoint(checkpoint_path_1)
    if latest_checkpoint_1:
        print(f"Restoring from checkpoint_1: {latest_checkpoint_1}")
        checkpoint_1.restore(latest_checkpoint_1)
    else:
        print("No checkpoint found. Starting from scratch.")

    model_loaded = model_1

    for layer in model_loaded.layers:
        print(layer.name)

    last_conv_layer = None


    for images, labels in ds_train.take(1):
            dummy_img = images[0:1]
            break
    _ = model_loaded(dummy_img)

    def find_target_layer(model):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(model.layers):
            # check to see if the layer has a 4D output
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name

    last_conv_layer_name = find_target_layer(model_loaded)

    if last_conv_layer:
        print(f"The last conv layer name is {last_conv_layer_name}")
    else:
        print(f"No Convolutional Layers found in the model")

    # Function to generate Grad-CAM
    def grad_cam(model, img_path,ds_train, last_conv_layer_name, target_class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Parameters:
          model: Trained model.
          img: Input image tensor (1, H, W, C).
          last_conv_layer_name (str): Name of the last convolutional layer.
          target_class_idx (optional): Target class index. If None, uses the predicted class.

        Returns:
          Heatmap normalized to the range [0, 1].
        """
        # Initialize the model,
        for images, labels in ds_train.take(1):
            dummy_img = images[0:1]
            break
        _ = model(dummy_img)

        # Load and preprocess the image
        image = cv2.imread(img_path) # Load image in BGR format

        if image is None:
            print(f"Error: File not found or cannot be opened at {img_path}")
        else:
            print("Image loaded successfully!")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        image = np.array(image) / 255.0  # Normalize the pixel values to [0, 1]
        img_array = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model(img_array, training=False)
        predictions = tf.sigmoid(predictions[:,0])
        print("Predicted class:", 1 if predictions[0] > 0.5 else 0)

        # Initialize a model for Grad-CAM
        grad_model = tf.keras.models.Model(
          inputs=model.input,
          outputs=[model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            # Ensure that the input tensor is watched
            tape.watch(img_array)

            # Get outputs for the last conv layer and the predictions
            conv_outputs, predictions = grad_model(img_array)

            # Extract the target class score
            class_output = predictions[:, 0]

        # Calculate gradients of the target class score with respect to conv layer output
        grads = tape.gradient(class_output, conv_outputs)
        print(grads.shape)

        # Compute the mean intensity of the gradients for each feature map
        weights = tf.reduce_mean(grads, axis=(0, 1, 2)) # along, B, H, W
        print(weights.shape)

        # Compute the weighted sum of the feature maps
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

        # Apply ReLU to retain only positive values
        cam = np.maximum(cam, 0)

        # Normalize the heatmap
        cam /= np.max(cam) if np.max(cam) != 0 else 1

        return cam.numpy()

    # Function to display Grad-CAM
    def display_grad_cam(heatmap, img, alpha=0.4):
        """
        Overlay Grad-CAM heatmap on the original image.

        Parameters:
          heatmap: Grad-CAM heatmap.
          img: Original image (H, W, C).
          alpha: Heatmap transparency factor.
        """
        # Convert heatmap to RGB
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Rescale original image to uint8
        img = np.uint8(img[0] * 255)

        # Overlay heatmap on original image
        overlayed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

        # Display images
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Grad-CAM Heatmap")
        plt.imshow(overlayed_img)
        plt.axis("off")
        plt.show()

    # Load and preprocess a test image
    img_path = r'F:\IDRID_dataset\images_augmented\images_augmented\test\binary\class_1\IDRiD_002.jpg'


    # Generate Grad-CAM heatmap
    heatmap = grad_cam(model_loaded, img_path, ds_train ,last_conv_layer_name)

    # Visualize the result
    display_grad_cam(heatmap, img_path)

if __name__ == '__main__':
    gin.parse_config_file(r'F:\dl lab\dl-lab-24w-team04-feature\Jiang241023\configs\config.gin')
    deep_visualization()