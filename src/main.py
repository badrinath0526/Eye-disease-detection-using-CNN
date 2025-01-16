# main.py

import tensorflow as tf
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
from modeltraining import create_model, train_and_evaluate
from visualization import visualize_images

# Load datasets directly without any preprocessing
train_ds = image_dataset_from_directory(
    directory='./Eyedisease',
    batch_size=16,
    shuffle=True,
    image_size=(256, 256),
    subset='training',
    seed=123,
    validation_split=0.2,
    label_mode='categorical',
)

validation_ds = image_dataset_from_directory(
    directory='./Eyedisease',
    batch_size=16,
    image_size=(256, 256),
    subset='validation',
    seed=123,
    validation_split=0.2,
    label_mode='categorical',
)

# Visualize a few images from the training set
# visualize_images('./Eyedisease/cataract', num_images=5)

# Create and compile the model
model = create_model()

# Train and evaluate the model
model = train_and_evaluate(model, train_ds, validation_ds, epochs=35)

# print(f"Test accuracy: {test_acc}")

# Save the trained model
# project_name = "Eye_Disease_Detection"
# model_filename = f"{project_name}.keras"
# model.save(model_filename)

# print(f"Model saved as {model_filename}")
