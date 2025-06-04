from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os
from io import BytesIO
from flask_cors import CORS
import pandas as pd  # Adicionado para ler o CSV

app = Flask(__name__)
CORS(app)

# Carrega o modelo e os labelsweb
modelo = load_model('models/modelo.h5')

# Carrega o mapeamento de classes
df_classes = pd.read_csv('./data/sports.csv')
class_mapping = dict(zip(df_classes['class id'], df_classes['labels']))

# Configurações
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(file_storage, target_size=(224, 224)):
    img = image.load_img(BytesIO(file_storage.read()), target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 
    file_storage.seek(0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhuma imagem enviada'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Nenhuma imagem selecionada'}), 400
    
    if file and allowed_file(file.filename):
        try:
            processed_image = prepare_image(file)
            predictions = modelo.predict(processed_image)[0]
            
            # Obtém os índices ordenados por probabilidade (do maior para menor)
            sorted_indices = np.argsort(predictions)[::-1]
            
            # Mapeia os índices para nomes de classes
            top_class = int(sorted_indices[0])
            top_class_name = class_mapping.get(top_class, "Classe desconhecida")
            top_prob = float(predictions[top_class])
            
            # Prepara as top 3 classes
            top3_classes = []
            top3_probs = []
            for i in sorted_indices[:3]:
                class_id = int(i)
                top3_classes.append({
                    'class_id': class_id,
                    'class_name': class_mapping.get(class_id, "Classe desconhecida"),
                })
                top3_probs.append(float(predictions[i]))
            
            # Formata a resposta
            response = {
                'top_class': {
                    'class_id': top_class,
                    'class_name': top_class_name,
                    'probability': top_prob
                },
                'top3_classes': top3_classes,
                'top3_probabilities': top3_probs
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Erro ao processar imagem: {str(e)}'}), 500
    
    else:
        return jsonify({'error': 'Formato de arquivo não permitido'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)