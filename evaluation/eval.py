import tensorflow as tf
from keras.utils.version_utils import training
from metrics import ConfusionMatrix
import logging
import numpy as np
import wandb


def evaluate(model_1, model_2, model_3, ds_test, ensemble = True):

    metrics = ConfusionMatrix()
    accuracy_list = []
    tp, fp, fn, tn = 0, 0, 0, 0

    for idx, (images, labels) in enumerate(ds_test):

        # Check the total number of test samples
        #test_total_samples = sum(1 for _ in ds_test.unbatch())
       # print(f"Total test samples in dataset: {test_total_samples}")

        threshold = 0.5

        # Model_1
        predictions_1 = model_1(images, training = False)
        predictions_1 = tf.cast(predictions_1 > threshold, tf.int32)

        # Model_2
        predictions_2 = model_2(images, training = False)
        predictions_2 = tf.cast(predictions_2 > threshold, tf.int32)

        # Model_3
        predictions_3 = model_3(images, training = False)
        predictions_3 = tf.cast(predictions_3 > threshold, tf.int32)

        # final_predictions
        votes = predictions_1 + predictions_2 + predictions_3  # Count votes (0, 1, or 2, or 3)
        if ensemble:
            final_predictions = tf.cast(votes > 1, tf.int32)
        else:
            final_predictions = predictions_2

        # Update accuracy
        matches = tf.cast(final_predictions == labels, tf.float32)
        batch_accuracy = tf.reduce_mean(matches)
        accuracy_list.append(batch_accuracy.numpy())

        # Update confusion matrix metrics
        metrics.update_state(labels, final_predictions)


    #print(f"Batch {idx}: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    # Calculate metrics
    accuracy = sum(accuracy_list) / len(accuracy_list)

    # Log the test accuracy to WandB
    wandb.log({'Test_accuracy': accuracy})

    total_samples = tp + fp + tn + fn
    #print(f"Total samples accounted for: {total_samples}")

    results = metrics.result()

    tp = results["tp"]
    fp = results["fp"]
    tn = results["tn"]
    fn = results["fn"]
    # update confusion matrix


    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    # Logging and printing results
    logging.info(f"Test_accuracy: {accuracy:.2%}")
    logging.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    logging.info(f"Sensitivity (Recall): {sensitivity:.2%}")
    logging.info(f"Specificity: {specificity:.2%}")
    logging.info(f"Precision: {precision:.2%}")
    logging.info(f"F1-Score: {f1_score:.2%}")

    print(f"Test_accuracy is: {accuracy:.2%}")
    print(f"Sensitivity (Recall): {sensitivity:.2%}")
    print(f"Specificity: {specificity:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"F1-Score: {f1_score:.2%}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    return {
        "Test_accuracy": accuracy,
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