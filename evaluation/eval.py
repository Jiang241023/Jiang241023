import tensorflow as tf
from metrics import ConfusionMatrix
import logging

def evaluate(model, ds_test, ds_info, run_paths, checkpoint = None):

    #if checkpoint:

    metrics = ConfusionMatrix()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
    evaluation_loss = tf.keras.metrics.Mean(name = 'evaluation_loss')
    evaluation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name = 'evaluation_accuracy')


    for test_images, test_labels in ds_test:
        predictions = model(test_images, training=False)
        loss = loss_object(test_labels, predictions)

        #update metrics
        evaluation_loss.update_state(loss)
        evaluation_accuracy.update_state(test_labels, predictions)
        metrics.update_state(y_true = test_labels, y_pred = predictions)

    final_evaluation_loss = evaluation_loss.result().numpy()
    final_evaluation_accuracy = evaluation_accuracy.result().numpy()

    logging.info(f"final_evaluation_loss: {final_evaluation_loss}")
    logging.info(f"final_evaluation_accuracy: {final_evaluation_accuracy}")


    return {
        "final_evaluation_loss": final_evaluation_loss,
        "final_evaluation_accuracy": final_evaluation_accuracy
    }