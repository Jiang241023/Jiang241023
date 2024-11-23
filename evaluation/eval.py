import tensorflow as tf
from keras.utils.version_utils import training
from tensorflow.python.layers.core import dropout
import numpy
from metrics import ConfusionMatrix
import logging

import os
test_data_dir = r"F:\IDRID_dataset\images_revized\test\binary"
class_0_count = len(os.listdir(os.path.join(test_data_dir, "class_0")))
class_1_count = len(os.listdir(os.path.join(test_data_dir, "class_1")))
print(f"Class 0: {class_0_count} images, Class 1: {class_1_count} images")

def evaluate(model, ds_test, ds_info, run_paths, checkpoint):

    # latest_checkpoint = tf.train.latest_checkpoint(run_paths["path_ckpts_train"])
    # if latest_checkpoint:
    #     checkpoint.rstore(latest_checkpoint)
    #     logging.info(f"Restored from the checkpoint: {latest_checkpoint}")
    # else:
    #     logging.info("No checkpoint found.")

    #metrics = ConfusionMatrix()
    accuracy_list = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for idx, (images, labels) in enumerate(ds_test):
        # Check the total number of test samples
        test_total_samples = sum(1 for _ in ds_test.unbatch())
        print(f"Total test samples in dataset: {test_total_samples}")

        predictions = model(images, training = False)
        threshold = 0.5
        predictions = tf.cast(predictions > threshold, tf.int32)

        # Update accuracy
        matches = tf.cast(predictions == labels, tf.float32)
        batch_accuracy = tf.reduce_mean(matches)
        accuracy_list.append(batch_accuracy.numpy())
        print(f"test Images shape: {images.shape}")
        print(f"test Labels shape: {labels.shape}")

        # Debug batch details
        print(f"Batch {idx}: Images shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Batch {idx}: Predictions shape: {predictions.shape}")

        # Check the sum of predictions and labels
        print(f"Batch {idx}: Sum of predictions: {tf.reduce_sum(predictions).numpy()}")
        print(f"Batch {idx}: Sum of labels: {tf.reduce_sum(labels).numpy()}")

        # Update confusion matrix metrics
        #metrics.update_state(labels, predictions)
        #print(f"true positive for a batch {idx} is {tf.cast((predictions == 1) & (labels == 1), tf.int32)}")
        tp += tf.reduce_sum(tf.cast((predictions == 1) & (labels == 1), tf.int32)).numpy()
        fp += tf.reduce_sum(tf.cast((predictions == 1) & (labels == 0), tf.int32)).numpy()
        tn += tf.reduce_sum(tf.cast((predictions == 0) & (labels == 0), tf.int32)).numpy()
        fn += tf.reduce_sum(tf.cast((predictions == 0) & (labels == 1), tf.int32)).numpy()

    #print(f"Batch {idx}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    # Calculate metrics
    accuracy = sum(accuracy_list) / len(accuracy_list)
    total_samples = tp + fp + tn + fn
    print(f"Total samples accounted for: {total_samples}")

    # results = metrics.result()
    #
    # tp = results["tp"]
    # fp = results["fp"]
    # tn = results["tn"]
    # fn = results["fn"]
    # update confusion matrix


    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Logging and printing results
    logging.info(f"Accuracy: {accuracy:.2%}")
    logging.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logging.info(f"Sensitivity (Recall): {sensitivity:.2%}")
    logging.info(f"Specificity: {specificity:.2%}")
    logging.info(f"Precision: {precision:.2%}")
    logging.info(f"F1-Score: {f1_score:.2%}")

    print(f"Accuracy is: {accuracy:.2%}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1-Score: {f1_score:.2%}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score,
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }
    }