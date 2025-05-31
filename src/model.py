import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações
DATA_DIR = './data'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30

# Caminhos das pastas
train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'valid')
test_dir = os.path.join(DATA_DIR, 'test')

# Geradores de dados
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Aumentado
    width_shift_range=0.3,  # Aumentado
    height_shift_range=0.3,  # Aumentado
    shear_range=0.3,  # Aumentado
    zoom_range=0.3,  # Aumentado
    horizontal_flip=True,
    vertical_flip=True,  # Adicionado
    brightness_range=[0.7, 1.3],  # Adicionado
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Carregando dados
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_test_datagen.flow_from_directory(
    validation_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Informações do modelo
num_classes = len(train_generator.class_indices)

model = Sequential([
    # Bloco 1
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.1),
    
    # Bloco 2
    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.1),
    
    # Bloco 3
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.1),
    
    # Bloco 4
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.1),
    
    # Classificador
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])

# SGD com momentum
# optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Callbacks
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=3)
]

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

model.compile(optimizer=Adam(learning_rate=0.0005),  # LR inicial menor
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=50,  # Mais épocas pois temos early stopping
    callbacks=[early_stop, reduce_lr]
)

# Avaliação
test_loss, test_acc = model.evaluate(test_generator)
print(f'\nAcurácia no teste: {test_acc:.2%}')

# Matriz de confusão
y_pred = np.argmax(model.predict(test_generator), axis=1)
cm = tf.math.confusion_matrix(test_generator.classes, y_pred)

# Salvando o modelo
model.save('models/modelo_2.h5')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.show()

# Gráfico de Acurácia
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()