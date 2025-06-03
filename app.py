from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os
from PIL import Image
import io

app = Flask(__name__)

# Carrega o modelo uma vez quando a API inicia
modelo = load_model('models/modelo.h5')

# Configurações
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img, target_size=(224, 224)):
    # Redimensiona a imagem para o tamanho alvo
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    
    # Converte para array numpy e pré-processa
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    # Verifica se a requisição tem a parte do arquivo
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    file = request.files['file']
    
    # Verifica se o arquivo tem um nome e extensão permitida
    if file.filename == '':
        return jsonify({'error': 'Nenhuma imagem selecionada'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Lê a imagem diretamente para memória
            img = Image.open(io.BytesIO(file.read()))
            
            # Prepara a imagem para classificação
            processed_image = prepare_image(img)
            
            # Faz a predição
            predictions = modelo.predict(processed_image)
            predictions = predictions[0]  # Pega o primeiro batch
            
            # Ordena as predições e pega os índices
            sorted_indices = np.argsort(predictions)[::-1]
            
            # Pega a classe com maior probabilidade
            top_class = int(sorted_indices[0])
            top_prob = float(predictions[top_class])
            
            # Pega as 3 classes com maior probabilidade
            top3_classes = [int(i) for i in sorted_indices[:3]]
            top3_probs = [float(predictions[i]) for i in sorted_indices[:3]]
            
            # Formata a resposta
            response = {
                'top_class': top_class,
                'top_prob': top_prob,
                'top3_classes': top3_classes,
                'top3_probs': top3_probs
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Erro ao processar imagem: {str(e)}'}), 500
    
    else:
        return jsonify({'error': 'Formato de arquivo não permitido'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)