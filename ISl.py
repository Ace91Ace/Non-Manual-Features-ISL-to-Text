# %% Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# %% Data loading and preprocessing
# Paths to the ISL dataset (update as per your local path)
train_dir = r'E:\CODE\Machine Learning\ISL\ISL_CSLRT_Corpus\Train'
valid_dir = r'E:\CODE\Machine Learning\ISL\ISL_CSLRT_Corpus\Validation'
test_dir = r'E:\CODE\Machine Learning\ISL\ISL_CSLRT_Corpus\Test'

# Image dimensions and batch size
img_width, img_height = 224, 224  # Input size for AlexNet
batch_size = 32  # Adjust based on memory capacity

# Data Augmentation and Loading
datagen = ImageDataGenerator(
    rescale=1. / 255,  # Normalize pixel values
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

valid_generator = datagen.flow_from_directory(
    valid_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)


# %% AlexNet architecture
def create_alexnet_model(input_shape):
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd, 4th, 5th Convolutional Layers
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # Flatten
    model.add(Flatten())

    # Fully Connected Layers
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))

    return model


# %% TimeDistributed and LSTM model
def create_sequential_model(sequence_length, input_shape):
    cnn_model = create_alexnet_model(input_shape)

    model = Sequential()

    # Apply TimeDistributed to the CNN
    model.add(TimeDistributed(cnn_model, input_shape=(sequence_length, img_width, img_height, 3)))

    # Add LSTM to process sequences of frames
    model.add(LSTM(64, return_sequences=False))

    # Output layer (adjust num_classes for your classification task)
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes for categorical classification

    return model


# %% Model compilation
sequence_length = 10  # Number of frames per sequence
input_shape = (img_width, img_height, 3)  # Shape for each frame

# Create the sequential model with TimeDistributed and LSTM
model = create_sequential_model(sequence_length, input_shape)

# Compile the model
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# %% Custom Data Generator: Create sequences of frames
def sequence_data_generator(generator, batch_size, sequence_length):
    """Yield batches of sequences of frames."""
    while True:
        x_batch, y_batch = [], []
        for _ in range(batch_size):
            x, y = next(generator)  # Use next() to get the next image and label
            # Collect a sequence of frames
            frames = [x]
            for _ in range(sequence_length - 1):  # Collect additional frames for the sequence
                x, _ = next(generator)
                frames.append(x)
            x_batch.append(np.stack(frames))  # Stack frames along the time axis
            y_batch.append(y)
        yield np.array(x_batch), np.array(y_batch)


# Adjusted generator with sequence data
train_sequence_generator = sequence_data_generator(train_generator, batch_size, sequence_length)
valid_sequence_generator = sequence_data_generator(valid_generator, batch_size, sequence_length)

# %% Train the model
epochs = 10  # Adjust based on model performance
steps_per_epoch = train_generator.samples // (batch_size * sequence_length)  # Steps based on sequences
validation_steps = valid_generator.samples // (batch_size * sequence_length)  # Validation steps

history = model.fit(
    train_sequence_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=valid_sequence_generator,
    validation_steps=validation_steps,
    epochs=epochs
)

# %% Evaluate the model on the test dataset
test_steps = test_generator.samples // batch_size  # Test steps
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# %% Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()
