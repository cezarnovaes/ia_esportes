import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Configurações
IMG_SIZE = (224, 224)
MODEL_PATH = 'models/modelo.h5'

# Carregar o modelo treinado
model = load_model(MODEL_PATH)

# Obter os nomes das classes (assumindo a mesma estrutura do treino)
class_names = sorted(os.listdir('./data/train'))  # Ajuste o caminho se necessário

def classify_image(img_path):
    """
    Classifica uma imagem usando o modelo treinado.
    
    Args:
        img_path (str): Caminho para a imagem a ser classificada
        
    Returns:
        dict: Dicionário com as classes e suas probabilidades
    """
    # Carregar e pré-processar a imagem
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalização

    # Fazer a predição
    predictions = model.predict(img_array)[0]
    
    # Retornar resultados formatados
    results = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Uso: python classify.py <caminho_para_imagem>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Erro: Arquivo '{image_path}' não encontrado.")
        sys.exit(1)
    
    print(f"\nClassificando imagem: {image_path}")
    results = classify_image(image_path)
    
    print("\nResultados da classificação:")
    for class_name, prob in results.items():
        print(f"{class_name}: {prob:.2%}")
    
    predicted_class = max(results, key=results.get)
    print(f"\nClasse predita: {predicted_class} ({(results[predicted_class]*100):.2f}% de confiança)")