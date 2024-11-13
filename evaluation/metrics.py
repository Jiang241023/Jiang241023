import tensorflow as tf

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init(self, num_classes, name="confusion_matrix", **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        # ...

    def update_state(self, y_true, y_pred, sample_weight=None):
        # ...

    def result(self):
        # ...
