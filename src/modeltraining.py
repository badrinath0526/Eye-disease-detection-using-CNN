# modeltraining.py

import tensorflow as tf
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, GlobalAveragePooling2D,MaxPooling2D,Conv2D,Dropout,Flatten
from keras._tf_keras.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Creates a model and compiles it
def create_model():
    model = Sequential([
        
        Conv2D(filters=32,kernel_size=(3,3),activation='relu',input_shape=(256,256,3),padding='same'),
        Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
        Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),

        Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),
        Conv2D(filters=128,kernel_size=(3,3),activation='relu',padding='same'),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(0.25),
        
        GlobalAveragePooling2D(),
        Flatten(),
        
        Dense(128, activation='relu'),  
        Dense(4, activation='softmax')
    ])
    

    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def train_and_evaluate(model, train_ds, validation_ds, epochs=5):
    earlystopping = EarlyStopping(monitor='val_loss',patience=3, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=[earlystopping])

    return model
