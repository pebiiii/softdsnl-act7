import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load IMDB dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)

# 2. Pad sequences
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=200)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200)

# 3. Build model
model = models.Sequential([
    layers.Embedding(10000, 32, input_length=200),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 4. Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Train
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test))

# 6. Save
model.save("imdb_text_model.h5")
print("✅ Model saved as imdb_text_model.h5")