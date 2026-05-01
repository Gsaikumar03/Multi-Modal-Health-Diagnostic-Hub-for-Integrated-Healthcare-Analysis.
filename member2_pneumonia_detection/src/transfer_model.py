import tensorflow as tf


def build_transfer_model(input_shape=(224, 224, 3)):

    inputs = tf.keras.Input(shape=input_shape)

    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    base_model.trainable = False

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model