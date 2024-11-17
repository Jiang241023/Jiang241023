import tensorflow as tf
from metrics import ConfusionMatrix
import logging


def evaluate(model, ds_test, ds_info, run_paths, checkpoint=None):
    metrics = ConfusionMatrix()
    accuracy_list = []

    # Reset the confusion matrix states before evaluation
    metrics.reset_states()

    for images, labels in ds_test:
        predictions = model(images, training=False)
        threshold = 0.6
        predictions = tf.cast(predictions > threshold, tf.int32)

        # Update accuracy
        matches = tf.cast(predictions == labels, tf.float32)
        batch_accuracy = tf.reduce_mean(matches)
        accuracy_list.append(batch_accuracy.numpy())

        # Update confusion matrix metrics
        metrics.update_state(labels, predictions)

    # Calculate metrics
    accuracy = sum(accuracy_list) / len(accuracy_list)
    results = metrics.result()

    tp = results["tp"]
    fp = results["fp"]
    tn = results["tn"]
    fn = results["fn"]

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

    print(f"Accuracy: {accuracy:.2%}")
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