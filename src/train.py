import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model import create_data_generators, build_model
import tensorflow as tf

# Configurações
EPOCHS = 50

# Obter geradores de dados e construir modelo
train_generator, validation_generator, test_generator = create_data_generators()
model = build_model(train_generator.num_classes)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=42,
        mode='max',
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        mode='max',
        verbose=1
    ),
    ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
]

# Calcular class weights
class_weights = compute_class_weight('balanced', 
                                   classes=np.unique(train_generator.classes), 
                                   y=train_generator.classes)
class_weight_dict = dict(enumerate(class_weights))

# Treinamento
print("\nIniciando treinamento...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    class_weight=class_weight_dict
)

model.save('models/modelo_2.h5')

# Avaliação completa
print("\nAvaliando modelo...")

# 1. Métricas
test_loss, test_acc = model.evaluate(test_generator, verbose=1)
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print(f'\nAcurácia no teste: {test_acc:.2%}')
print(f"Precisão (weighted): {precision:.2%}")
print(f"Recall (weighted): {recall:.2%}")
print(f"F1-Score (weighted): {f1:.2%}")

# 2. Métricas detalhadas
print("\nRelatório de Classificação Completo:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

# 3. ROC-AUC
try:
    roc_auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
    print(f"ROC-AUC Score (One-vs-Rest): {roc_auc:.2%}")
except Exception as e:
    print(f"\nNão foi possível calcular ROC-AUC: {str(e)}")

# 4. Matriz de confusão
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(test_generator.class_indices.keys()),
            yticklabels=list(test_generator.class_indices.keys()))
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão Detalhada')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# 5. Gráficos de desempenho
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia durante o Treino')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss durante o Treino')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()