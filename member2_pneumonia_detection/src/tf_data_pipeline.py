import tensorflow as tf


IMG_SIZE = 224
BATCH_SIZE = 32


def create_datasets(data_dir: str):
    """
    Create TensorFlow datasets from directory
    """

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + "/train",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + "/val",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir + "/test",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    return train_ds, val_ds, test_ds


def apply_normalization(train_ds, val_ds, test_ds):
    """
    Normalize pixel values to [0,1]
    """

    normalization_layer = tf.keras.layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, val_ds, test_ds


def apply_augmentation(train_ds):
    """
    Apply data augmentation only on training set
    """

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
    return train_ds
