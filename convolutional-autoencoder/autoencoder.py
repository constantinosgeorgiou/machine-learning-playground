from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import numpy as np

import gzip

from ignored import inputs


def extract_data(filename, images):
    """
    Extract images from gzip file
    """

    with gzip.open(filename) as bytestream:
        bytestream.read(16)

        buffer = bytestream.read(28*28*images)

        # Convert buffer to NumPy array of type float32
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)

        # Reshape into 3D array (tensor)
        data = data.reshape(images, 28, 28)

        return data


def extract_labels(filename, images):
    """
    Extract labels from gzip file
    """

    with gzip.open(filename) as bytestream:
        bytestream.read(8)

        buffer = bytestream.read(1*images)

        # Convert buffer to NumPy array of type int64
        labels = np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)

        return labels


def autoencoder(image):
    """
    Autoencoder description
    """

    # Encoder
    # -------
    # input: 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(
        32, (3, 3), activation="relu",
        padding="same"
    )(image)  # 28 x 28 x 32
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # 14 x 14 x 32

    conv2 = Conv2D(
        63, (3, 3), activation="relu", padding="same"
    )(pool1)  # 14 x 14 x 64
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # 7 x  7 x 64

    conv3 = Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(pool2)  # 7 x 7 x 128 (small and thick)

    # Decoder
    # -------
    # input: 7 x 7 x 128 (small and thick)
    conv4 = Conv2D(
        128, (3, 3), activation="relu", padding="same"
    )(conv3)  # 7 x 7 x 128
    up1 = UpSampling2D((2, 2))(conv4)  # 14 x 14 x 128

    conv5 = Conv2D(
        64, (3, 3), activation="relu", padding="same"
    )(up1)  # 14 x 14 x 64
    up2 = UpSampling2D((2, 2))(conv5)  # 28 x 28 x 64

    decoded = Conv2D(
        1, (3, 3), activation="sigmoid", padding="same"
    )(up2)  # 28 x 28 x 1 (wide and thin)

    return decoded


if __name__ == "__main__":
    train_data_input = inputs.train_data_input
    test_data_input = inputs.test_data_input

    train_labels_input = inputs.train_labels_input
    test_labels_input = inputs.test_labels_input

    train_data = extract_data(train_data_input, 60000)
    test_data = extract_data(test_data_input, 10000)

    train_labels = extract_labels(train_labels_input, 60000)
    test_labels = extract_labels(test_labels_input, 10000)

    # Data preprocessing
    # ------------------
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)

    # Shapes of training set
    print(
        "Training set (images) shape: ",
        train_data.shape, "dtype: ", train_data.dtype
    )

    # Shapes of test set
    print(
        "Test set (images) shape: ",
        test_data.shape, "dtype: ", test_data.dtype
    )

    # Rescale the training and testing data with the maximum pixel value
    print(train_data.shape, np.max(train_data))
    print(test_data.shape, np.max(test_data))
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)
    print(train_data.shape, np.max(train_data))
    print(test_data.shape, np.max(test_data))

    # Partition data into training set and validation set
    # 80% data used for training
    # 20% data used for validation
    # Why? Reduce the chances of overfitting
    train_X, valid_X, train_ground, valid_ground = train_test_split(
        train_data, train_data, test_size=0.2, random_state=13
    )

    batch_size = 128
    epochs = 50
    inChannel = 1
    x, y = 28, 28
    image = Input(shape=(x, y, inChannel))

    autoencoder = Model(image, autoencoder(image))

    autoencoder.compile(loss="mean_squared_error", optimizer=RMSprop())

    autoencoder.summary()

    autoencoder_train = autoencoder.fit(
        train_X, train_ground, batch_size=batch_size, epochs=epochs,
        verbose=1, validation_data=(valid_X, valid_ground)
    )

    loss = autoencoder_train.history("loss")

    val_loss = autoencoder_train.history("val_loss")

    epochs = range(epochs)

    plt.figure()

    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
