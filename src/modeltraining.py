# modeltraining.py

import tensorflow as tf
from keras._tf_keras.keras.applications import EfficientNetB3
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D,MaxPooling2D,Conv2D,Dropout,BatchNormalization
from keras._tf_keras.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def create_model():
    # base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(256,256,3))
    model = Sequential([
        
        Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(256,256,3),padding='same'),
        # BatchNormalization(),
        Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'),
        # BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
        # BatchNormalization(),
        Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
        # BatchNormalization(),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),  
        Dense(4, activation='softmax')
    ])
    
    # for layer in base_model.layers:
    #     layer.trainable = False  # Freeze the layers of the base model
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def train_and_evaluate(model, train_ds, validation_ds, epochs=5):
    earlystopping = EarlyStopping(monitor='val_loss',patience=3, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=[earlystopping])

    return model
