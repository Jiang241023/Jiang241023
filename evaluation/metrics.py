import tensorflow as tf


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

    def update_state(self, y_true, y_pred, sample_weight = None):
        # ...


        # calculate the confusion matrix
        result = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.num_classes, dtype = tf.int32)

        # update the existing confusion matrix by adding the values to it
        self.confusion_matrix.assign_add(result)


    def result(self):
        # ...
        return self.confusion_matrix