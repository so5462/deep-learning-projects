import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class CustomDataWrapper():
    def __init__(self, dataset):
        batch_size = 32

        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        x_train, x_test = x_train/255.0, x_test/255.0

        # Use tf.data to batch and shuffle the ds
        dataset_train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(1024).batch(batch_size)

        dataset_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        plt.figure()
        plt.gray()
        plt.imshow(a[0])
        plt.show()



"""
Start testing the classes here
"""
if __name__ == '__main__':
    # mnist = tf.keras.datasets.mnist
    # dataWrapper = CustomDataWrapper(mnist)
    unit_vector = tf.eye(100)[1, :]
    x = np.zeros((100, 10))
    x = tf.cast(x, tf.float32)
    b  = tf.multiply(x, unit_vector)
    print(b)







