"""
Tensorflow Tutorial
Quickstart for beginners

Outcome: Short introduction to Keras
"""

import tensorflow as tf

print(f'TensorFlow version: {tf.__version__}')

# Load a dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a machine learning model by stacking sequential layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# The model returns a vector of logits or log-odds scores, one for each class
predictions = model(x_train[:1]).numpy()
print(predictions)

# Convert logits to probabilities for each class
tf.nn.softmax(predictions).numpy()

# Define a loss function
"""
This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
"""
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Test the model on a test/validation set
model.evaluate(x_test,  y_test, verbose=2)