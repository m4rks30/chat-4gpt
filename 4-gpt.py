import tensorflow as tf
import tensorflow_datasets as tfd
import os

# Load the dataset
dataset, info = tfd.load('conv_ai/2', with_info=True)
train_data = dataset['train']
validation_data = dataset['validation']

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=8000),
    tf.keras.layers.Embedding(8000, 64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data,
          validation_data=validation_data,
          epochs=10)

# Save the model
model.save('chatbot_gpt')
