# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 12:59:12 2025

@author: vikas
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.datasets import cifar10
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from keras.callbacks import CSVLogger

# Hyperparameters
BATCH_SIZE = 256
EPOCHS = 200
size=32
INPUT_SHAPE = (size, size, 3)

(x_train, y_train), (x_test, y_test), (x_val, y_val) = (np.load("FData/X_train.npy"),np.load("FData/y_train.npy")),(np.load("FData/X_test.npy"),np.load("FData/y_test.npy")),(np.load("FData/X_val.npy"),np.load("FData/y_val.npy"))
NUM_CLASSES = y_train.shape[1]
x_train.shape

# Function to create a simple student model
def create_student_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Simple CNN architecture
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, outputs)

# Function to create teacher model based on architecture name
def create_teacher_model(model_name, input_shape, num_classes):
    # Resize input for models that expect larger input sizes
    inputs = keras.Input(shape=input_shape)
    x = layers.Lambda(lambda image: tf.image.resize(image, (size, size)))(inputs)
    if model_name == 'vgg16':
        base_model = applications.VGG16(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'vgg19':
        base_model = applications.VGG19(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'resnet':
        base_model = applications.ResNet50(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'densenet':
        base_model = applications.DenseNet121(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'efficientnet':
        base_model = applications.EfficientNetB3(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'nasnetmobile':
        base_model = applications.NASNetMobile(weights='imagenet', include_top=False, input_tensor=x)
    elif model_name == 'xception':
        base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor=x)
    else:
        raise ValueError("Unknown model name")
    
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    model = keras.Model(base_model.input, outputs)
    
    
    return model

# Custom distillation loss function
class Distiller(keras.Model):
    def __init__(self, student, teacher):
        self.teacher = teacher
        self.student = student

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn):
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.distillation_loss_fn = distillation_loss_fn

    def train_step(self, data):
        x, y = data
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)
            
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            
            # Compute distillation loss with temperature
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / TEMPERATURE, axis=1),
                tf.nn.softmax(student_predictions / TEMPERATURE, axis=1)
            )
            
        
        # Compute gradients and update student weights
        trainable_vars = self.student.trainable_variables
        # Update the metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "student_loss": student_loss,
            "distillation_loss": distillation_loss,
        })
        return results

    def test_step(self, data):
        x, y = data
        
        # Forward pass of student
        student_predictions = self.student(x, training=False)
        
        # Calculate the loss
        student_loss = self.student_loss_fn(y, student_predictions)
        
        # Update the metrics
        self.compiled_metrics.update_state(y, student_predictions)
        
        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results

# Function to train and evaluate knowledge distillation
def train_knowledge_distillation(teacher_model_name):
    print(f"\nTraining with {teacher_model_name} as teacher")
    
    # Create teacher and student models
    teacher = create_teacher_model(teacher_model_name, INPUT_SHAPE, NUM_CLASSES)
    student = create_student_model(INPUT_SHAPE, NUM_CLASSES)
    
    # Compile teacher (only top layers are trainable)
    teacher.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    print(teacher.summary())
    
    csv_logger = CSVLogger(f'Results//{teacher_model_name}_teachermodel.csv')
    # Train teacher on the data (only top layers)
    print("Training teacher model...")
    teacher.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[csv_logger]
    )
    
    teacher.evaluate(x_val, y_val, verbose=2)
    y_predT = student.predict(x_val)
    
    # Initialize and compile distiller
    csv_logger = CSVLogger(f'Results//{teacher_model_name}_teachermodel.csv')
    distiller = Distiller(student=student, teacher=teacher)
    distiller.compile(
        optimizer=optimizers.Adam(),
        metrics=['accuracy', 'Precision', 'Recall'],
        student_loss_fn=keras.losses.CategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
    )
    print(distiller.summary())
    # Distill teacher to student
    print("Distilling knowledge to student model...")
    #csv_logger = CSVLogger(f'Results//{teacher_model_name}_Distillermodel.csv')
    history = distiller.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        #callbacks=[csv_logger]
    )
    
    # Evaluate student standalone
    print("Evaluating student model...")
    student.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'Precision', 'Recall']
    )
    print(student.summary())
    student.evaluate(x_val, y_val, verbose=2)
    y_pred = student.predict(x_val)
    return student, history, y_pred, y_predT

# Train with different teacher models
teacher_models = ['vgg16', 'vgg19', 'resnet', 'densenet', 'efficientnet',
                  'nasnetmobile','xception']
for model_name in teacher_models:
    #try:
        student_model, history, y_pred, y_predT = train_knowledge_distillation(model_name)
        # Classification report
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(np.argmax(y_val, axis=1), y_pred_classes, output_dict=True)
         
    #except Exception as e:
        #print(f"Error with {model_name}: {str(e)}")
    