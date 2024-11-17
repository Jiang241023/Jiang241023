import tensorflow as tf
from metrics import ConfusionMatrix
import logging

def evaluate(model, ds_test, ds_info, run_paths, checkpoint = None):

    #if checkpoint:

    metrics = ConfusionMatrix()

    evaluation_accuracy = tf.keras.metrics.BinaryAccuracy(name = 'evaluation_accuracy')


    for test_images, test_labels in ds_test:

        #print(f"test image shape: {test_images.shape}")
        #print(f"test_labels shape: {test_labels.shape}")
        predictions = model(test_images, training=False)
        predictions = tf.squeeze(predictions, axis=-1)
        #print(f"prediction shape: {predictions.shape}")

        # convert the prediction to 0 to 1 using threshold = 0.5
        predictions = tf.cast(predictions > 0.5, tf.int32)
        # accuracy --> compare prediction and true labels,

        #update metrics
        evaluation_accuracy.update_state(test_labels, predictions)
        metrics.update_state(y_true = test_labels, y_pred = predictions)

        

    final_evaluation_accuracy = evaluation_accuracy.result().numpy()
    confusion_matrix_result = metrics.result().numpy()

    logging.info(f"final_evaluation_accuracy: {final_evaluation_accuracy:.2%}")
    logging.info(f"confusion_matrix_result: {confusion_matrix_result}")

    nums_TP = confusion_matrix_result[1, 1]
    nums_TN = confusion_matrix_result[0, 0]
    nums_FP = confusion_matrix_result[0, 1]
    nums_FN = confusion_matrix_result[1, 0]


    print(
        f" the number of TPs: {nums_TP}\n",
        f"the number of TNs: {nums_TN}\n",
        f"the number of FPs: {nums_FP}\n",
        f"the number of FNs: {nums_FN}"
    )
    return {
        "final_evaluation_accuracy:", final_evaluation_accuracy,
        f"the number of TPs:{nums_TP}\n",
        f"the number of TNs:{nums_TN}\n",
        f"the number of FPs:{nums_FP}\n",
        f"the number of FNs:{nums_FN}"
    }
