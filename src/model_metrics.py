import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_score, recall_score, f1_score, 
                            roc_auc_score, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_model(model_path, test_data_dir, img_size=(224, 224), batch_size=32):
    """
    Analisa um modelo salvo (.h5) e retorna métricas de desempenho nos dados de teste.
    
    Args:
        model_path (str): Caminho para o arquivo .h5 do modelo
        test_data_dir (str): Diretório contendo os dados de teste
        img_size (tuple): Tamanho das imagens (altura, largura)
        batch_size (int): Tamanho do batch para avaliação
        
    Returns:
        dict: Dicionário com todas as métricas calculadas
    """
    
    # 1. Carregar o modelo
    print(f"\nCarregando modelo de {model_path}...")
    model = load_model(model_path)
    model.summary()
    
    # 2. Preparar gerador de dados de teste
    print("\nPreparando dados de teste...")
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # 3. Avaliação básica
    print("\nAvaliando modelo...")
    test_loss, test_acc = model.evaluate(test_generator, verbose=1)
    print(f"\nAcurácia no teste: {test_acc:.4f}")
    print(f"Loss no teste: {test_loss:.4f}")
    
    # 4. Previsões para métricas adicionais
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes
    
    # 5. Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')
    except Exception as e:
        print(f"\nNão foi possível calcular ROC-AUC: {str(e)}")
        metrics['roc_auc'] = None
    
    # 6. Relatório de classificação
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # 7. Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # 8. Salvar resultados
    results = {
        'basic_metrics': {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc)
        },
        'additional_metrics': metrics,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    # 9. Exibir resumo
    print("\n=== Resumo das Métricas ===")
    print(f"Acurácia: {metrics['accuracy']:.4f}")
    print(f"Precisão: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisar modelo .h5 com dados de teste')
    parser.add_argument('model_path', type=str, help='Caminho para o arquivo .h5 do modelo')
    parser.add_argument('test_data_dir', type=str, help='Diretório com dados de teste')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], 
                        help='Tamanho das imagens (altura largura)')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Tamanho do batch para avaliação')
    
    args = parser.parse_args()
    
    results = analyze_model(
        model_path=args.model_path,
        test_data_dir=args.test_data_dir,
        img_size=tuple(args.img_size),
        batch_size=args.batch_size
    )
    
    # Salvar resultados em JSON
    import json
    with open('model_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\nAnálise concluída. Resultados salvos em 'model_metrics.json' e 'confusion_matrix.png'")