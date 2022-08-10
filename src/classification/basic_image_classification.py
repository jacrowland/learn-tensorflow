"""
Tensorflow Tutorial
Basic classification: Classify images of clothing

Outcome: Train a nueral network model to classify images of clothing i.e. sneakers and shirts
"""

from pickletools import optimize
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# Load the Fashion Mnist dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()  # labels are 0 through 9

# Define English class names e.g. class 0 = T-shirt/top
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Training images dataset shape
print(train_images.shape)
print(len(train_labels))
print(train_labels)

# Testing images dataset shape
print(test_images.shape)
print(len(test_images))
print(test_labels)

# Preprocess the data

def plot_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

# Show first image
plot_image(train_images[0])

# Scale images to between 0 and 255
train_images = train_images / 255.0
test_images = test_images / 255.0

# Plot and display class of the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Build the model

# Define the layers
model = tf.keras.Sequential([
    # input layer that transforms the format of the images from 2D array to 1D array (28 * 28 = 784 pixels)
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # middle layer of 128 neurons
    tf.keras.layers.Dense(128, activation='relu'),
    # output layer returning logits array length 10 containing the score that each iamge belongs to one of the 10 classes
    tf.keras.layers.Dense(10)
])

# Compile the model
"""
    To compile we must define:
    - Loss function
        Measures how accurate the model is during training. You want to minmize this function to "steer" the model in the right direction (-> gradient descent)
    - Optimizer
        How the model is updated based on the data it sees and its loss function
    - Metrics
        Used to monitor the training and testing steps. The following example uses accuracy (fraction of images correctly classified)
"""
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True), metrics=['accuracy'])

# Fit and test the model
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Predict
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[np.argmax(predictions[0])])


def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])