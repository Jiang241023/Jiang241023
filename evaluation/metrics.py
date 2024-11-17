import tensorflow as tf
from wandb.integration.sklearn.plot.classifier import confusion_matrix


class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self, num_classes = 2, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name = name, **kwargs)
        # ...
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name = 'confusion_matrix',
            shape = (self.num_classes,  self.num_classes),
            initializer = 'zeros',
            dtype = tf.int32
        )
        #self.tp = self.add_weight(name = 'true positives', initializer = 'zeros',)
    def update_state(self, y_true, y_pred, sample_weight = None):
        # Cast inputs to integer
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # Calculate components of the confusion matrix
        tp = tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 1), tf.int32))
        fp = tf.reduce_sum(tf.cast((y_pred == 1) & (y_true == 0), tf.int32))
        tn = tf.reduce_sum(tf.cast((y_pred == 0) & (y_true == 0), tf.int32))
        fn = tf.reduce_sum(tf.cast((y_pred == 0) & (y_true == 1), tf.int32))

        # Convert the confusion matrix to a Tensor
        confusion_matrix = tf.convert_to_tensor(self.confusion_matrix)

        # Update the confusion matrix
        updated_confusion_matrix = tf.tensor_scatter_nd_add(
            confusion_matrix,
            indices=[[1, 1], [0, 1], [0, 0], [1, 0]],  # TP, FP, TN, FN indices
            updates=[tp, fp, tn, fn]
        )

        # Assign the updated confusion matrix to the weight
        self.confusion_matrix.assign(updated_confusion_matrix)


    def result(self):
        # ...
        return self.confusion_matrix

    def reset_states(self):
        # Reset the confusion matrix to zeros
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))



