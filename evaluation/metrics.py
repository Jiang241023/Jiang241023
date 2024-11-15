import gin
import tensorflow as tf

@gin.configurable
class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...
        num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name = 'confusion_matrix',
            shape = (num_classes, num_classes),
            initializer = 'zeros',
            dtype = tf.int32
        )

    def update_state(self, y_true, y_pred, sample_weight = None):
        # ...
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # calculate the confusion matrix
        result = tf.math.confusion_matrix(y_true, y_pred, num_classes = self.confusion_matrix, dtype = tf.int32)

        # update the existing confusion matrix by adding the values to it
        self.confusion_matrix.assign_add(result)


    def result(self):
        # ...
        return self.confusion_matrix