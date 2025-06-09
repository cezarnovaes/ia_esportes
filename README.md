# 🏆 ia_esportes - Sistema Inteligente de Classificação de Esportes

Um sistema completo de classificação de imagens esportivas utilizando **Deep Learning** e **Transfer Learning** com MobileNetV2, incluindo interface web interativa, API REST e pipeline completo de treinamento e avaliação.

## 🎯 Visão Geral do Projeto

Este projeto implementa uma solução end-to-end para classificação automática de modalidades esportivas em imagens, utilizando técnicas avançadas de inteligência artificial. O sistema combina a eficiência do MobileNetV2 com estratégias otimizadas de transfer learning para alcançar alta performance na identificação de diferentes esportes.

### ✨ Características Principais

- 🧠 **Transfer Learning** com MobileNetV2 pré-treinada no ImageNet
- 🔄 **Data Augmentation** inteligente para robustez do modelo
- 📊 **Interface Web** interativa para demonstração e testes
- 🚀 **API REST** para integração com outras aplicações
- 📈 **Métricas detalhadas** de avaliação e visualizações
- ⚡ **Pipeline otimizado** de treinamento com callbacks adaptativos

## 📁 Estrutura do Projeto

```
ia_esportes/
├── data/
│   ├── train/           # Dados de treinamento organizados por classe
│   ├── valid/           # Dados de validação para monitoramento
│   └── test/            # Dados de teste para avaliação final
├── models/              # Modelos treinados (.h5)
├── src/
│   ├── img/             # Recursos visuais (gráficos, ícones)
│   ├── web/             # Interface web completa
│   │   ├── index.html   # Página principal com demonstração
│   │   ├── script.js    # Lógica de interação e integração com API
│   │   └── styles.css   # Estilos modernos e responsivos
│   ├── model.py         # Arquitetura e construção do modelo
│   ├── train.py         # Pipeline de treinamento e avaliação
│   ├── predict.py       # Sistema de predição para novas imagens
│   └── api.py           # API Flask para classificação em tempo real
├── requirements.txt     # Dependências do projeto
└── README.md           # Documentação completa
```

## 🏗️ Arquitetura do Modelo

### Fluxo de Processamento
```
Imagem (224x224x3) → MobileNetV2 → Global Average Pooling → 
Dropout + BatchNorm → Dense Layer → Classificação Final
```

### Configurações Técnicas
- **Modelo Base**: MobileNetV2 (ImageNet pré-treinada)
- **Resolução**: 224x224 pixels (otimizada para MobileNetV2)
- **Batch Size**: 32
- **Learning Rate**: 1e-4 com decaimento exponencial
- **Épocas**: 50 (com early stopping inteligente)
- **Regularização**: Dropout (20%) + Batch Normalization + Weight Decay

### Estratégia de Transfer Learning
1. **Congelamento**: Todas as camadas exceto as últimas 20
2. **Fine-tuning**: Adaptação gradual ao domínio esportivo
3. **Otimização**: Adam com weight decay (1e-4)
4. **Loss Function**: Categorical Crossentropy com label smoothing (0.1)

## 🔧 Tecnologias Utilizadas

### Core ML/AI
- **TensorFlow 2.x** - Framework principal de deep learning
- **Keras** - API de alto nível para construção do modelo
- **MobileNetV2** - Arquitetura eficiente para transfer learning
- **NumPy** - Computação numérica e manipulação de arrays
- **Scikit-learn** - Métricas de avaliação e análise

### Visualização e Análise
- **Matplotlib** - Gráficos de treinamento e métricas
- **Seaborn** - Visualizações estatísticas avançadas
- **Plotly** - Gráficos interativos (opcional)

### Web e API
- **Flask** - Framework web para API REST
- **HTML5/CSS3/JavaScript** - Interface web moderna
- **Font Awesome** - Ícones e elementos visuais

### Processamento de Dados
- **Pillow (PIL)** - Manipulação e processamento de imagens
- **OpenCV** - Operações avançadas de visão computacional (opcional)

## 🚀 Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- GPU compatível com CUDA (recomendado)
- 8GB+ RAM
- 2GB+ espaço em disco

### Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/ia_esportes.git
cd ia_esportes

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\\Scripts\\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

## 📊 Treinamento do Modelo

### Preparação dos Dados
O sistema implementa data augmentation conservadora para preservar características essenciais:

```python
# Configuração de Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)
```

### Execução do Treinamento
```bash
# Treinamento completo com avaliação
python src/train.py

# Monitoramento em tempo real (opcional)
tensorboard --logdir=logs/
```

### Callbacks Inteligentes
- **EarlyStopping**: Parada antecipada (patience=10)
- **ReduceLROnPlateau**: Redução adaptativa do learning rate
- **ModelCheckpoint**: Salvamento do melhor modelo
- **CSVLogger**: Log detalhado do treinamento

## 🎯 Avaliação e Métricas

O sistema gera análises completas de performance:

### Métricas Implementadas
- ✅ **Acurácia** - Performance geral do modelo
- ✅ **Precisão** - Qualidade das predições positivas
- ✅ **Recall** - Capacidade de detectar casos positivos
- ✅ **F1-Score** - Média harmônica entre precisão e recall
- ✅ **ROC-AUC** - Análise multiclasse da curva ROC
- ✅ **Matriz de Confusão** - Distribuição detalhada de erros

### Visualizações Geradas
- 📈 Histórico de treinamento (loss e accuracy)
- 🎯 Matriz de confusão normalizada
- 📊 Relatório de classificação por classe
- 🔄 Curvas de aprendizado

## 🌐 Interface Web e API

### Demonstração Interativa
A interface web oferece:
- 📤 **Upload de imagens** com drag & drop
- ⚡ **Classificação em tempo real**
- 📊 **Visualização de confiança** por classe
- 🎨 **Design responsivo** e moderno

### API REST
```bash
# Iniciar servidor da API
python src/api.py

# Endpoint de classificação
POST /predict
Content-Type: multipart/form-data
Body: image file
```

### Exemplo de Uso da API
```python
import requests

# Classificar uma imagem
with open('imagem_esporte.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:5000/predict',
        files={'image': f}
    )
    
resultado = response.json()
print(f"Esporte detectado: {resultado['classe']}")
print(f"Confiança: {resultado['confianca']:.2%}")
```

## 🔮 Predição em Novas Imagens

```bash
# Classificar uma imagem específica
python src/predict.py --image caminho/para/imagem.jpg

# Classificar múltiplas imagens
python src/predict.py --folder caminho/para/pasta/
```

## 📈 Resultados e Performance

### Configurações de Treinamento
| Parâmetro | Valor |
|-----------|-------|
| Épocas Máximas | 50 |
| Batch Size | 32 |
| Learning Rate Inicial | 1e-4 |
| Resolução de Entrada | 224x224 |
| Dropout Rate | 0.2 |
| Weight Decay | 1e-4 |

### Otimizações Implementadas
- 🎯 **Fine-tuning seletivo** das últimas 20 camadas
- 🔄 **Learning rate scheduling** adaptativo
- ⚖️ **Class weighting** para datasets desbalanceados
- 🛡️ **Regularização multicamada** para prevenir overfitting

## 🚀 Próximos Passos

### Melhorias Planejadas
- [ ] Implementação de ensemble de modelos
- [ ] Suporte a vídeos e classificação temporal
- [ ] API GraphQL para consultas flexíveis
- [ ] Deploy automatizado com Docker
- [ ] Monitoramento de drift de dados
- [ ] Explicabilidade com Grad-CAM

### Expansões Futuras
- [ ] Detecção de objetos esportivos específicos
- [ ] Análise de performance de atletas
- [ ] Integração com streaming de vídeo
- [ ] Aplicativo mobile nativo