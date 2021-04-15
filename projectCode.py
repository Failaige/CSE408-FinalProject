import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# preparing CIFAR10 dataset
# for the project it might be best for us to modify the test images to the corrupted ones, not entirely sure how
(trainImages, trainLabels), (testImages, testLabels) = datasets.cifar10.load_data()
# normalize pixel values to between 0 and 1
trainImages = trainImages / 255.0
testImages = testImages / 255.0

# list of class names
classNames = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# data verification
# This will plot the first 25 images in the dataset with their labels (not necessary for testing)
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap = plt.cm.binary)
    plt.xlabel(classNames[trainLabels[i][0]])
# shows the plot
plt.show()

# convolutional base
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# displays the architecture of the model so far
model.summary()

# dense layers added to top of model to perform classification
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# displays the completed model
model.summary()

# compilation and training of the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(trainImages, trainLabels, epochs=10, validation_data=(testImages, testLabels))

# evaluating the model
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss = model.evaluate(testImages, testLabels, verbose=2)
test_acc = model.evaluate(testImages, testLabels, verbose=2)
print(test_acc)
