import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math 

(data, ds_info) = tfds.load('fashion_mnist', as_supervised=True, with_info=True)


training_data, test_data = data['train'], data['test']
class_names = ds_info.features['label'].names

print(class_names)

def normalize(images, tags): 
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, tags

training_data = training_data.map(normalize)
test_data = test_data.map(normalize)

training_data = training_data.cache()
test_data = test_data.cache()

for image, tag in training_data.take(1):
    break
image = image.numpy().reshape((28,28))

# plt.figure()
# plt.imshow(image, cmap=plt.cm.binary)
# plt.colorbar()
# plt.grid(False)
# plt.show()

#creating the model

model  = tf.keras.Sequential([
tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28, 28, 1)),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D((2,2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer = 'adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

num_training = ds_info.splits['train'].num_examples
num_test = ds_info.splits['test'].num_examples

batch_size = 32

training_data = training_data.repeat().shuffle(num_training).batch(batch_size)
test_data = test_data.batch(batch_size)

history = model.fit(training_data, epochs=5, steps_per_epoch=math.ceil(num_training/batch_size))

plt.xlabel("# Epoch")
plt.ylabel("magnitude of loss")
plt.plot(history.history["loss"])
plt.show()