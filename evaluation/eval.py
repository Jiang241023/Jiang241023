import tensorflow as tf
from metrics import ConfusionMatrix
import logging

def evaluate(model, ds_test, ds_info, run_paths, checkpoint=None):
    metrics = ConfusionMatrix()
    evaluation_accuracy = tf.keras.metrics.BinaryAccuracy(name='evaluation_accuracy')

    for test_images, test_labels in ds_test:
        predictions = model(test_images, training=False)
        predictions = tf.squeeze(predictions, axis=-1)
        threshold = 0.39
        predictions = tf.cast(predictions > threshold , tf.int32)


        evaluation_accuracy.update_state(test_labels, predictions)
        metrics.update_state(test_labels, predictions)


    confusion_matrix_result = metrics.result().numpy()
    final_evaluation_accuracy = evaluation_accuracy.result().numpy()


    tp = confusion_matrix_result[1, 1]
    fp = confusion_matrix_result[0, 1]
    tn = confusion_matrix_result[0, 0]
    fn = confusion_matrix_result[1, 0]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 召回率
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异性
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # 精确率
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    logging.info(f"final_evaluation_accuracy: {final_evaluation_accuracy:.2%}")
    logging.info(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    print(
        f"Accuracy: {final_evaluation_accuracy:.2%}\n",
        f"Sensitivity (Recall): {sensitivity:.2%}\n",
        f"Specificity: {specificity:.2%}\n",
        f"Precision: {precision:.2%}\n",
        f"F1-Score: {f1_score:.2%}\n",
        f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}"
    )

    return {
        "accuracy": final_evaluation_accuracy,
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