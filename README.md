# ia_esportes

Este projeto implementa um sistema completo de classificação de imagens usando transfer learning com MobileNetV2, incluindo treinamento, avaliação e predição com uma api e página web.

## Visão Geral do Projeto

ia_esportes/
├── data/
│   ├── train/       # Dados de treino
│   ├── valid/       # Dados de validação
│   └── test/        # Dados de teste
├── models/          # Modelos treinados
├── src/
│   ├── img/         # Imagens da página
│   ├── web/         # Código da página web
│   ├── model.p      # Construção do modelo
│   ├── train.py     # Script de treinamento
│   ├── predict.py   # Script de predição
│   ├── api.py       # Código da api
├── requirements.txt # Dependências
└── README.md        # Este arquivo

O sistema consiste em três componentes principais do modelo, dois componentes da parte web e a api:
1. `model.py` - Construção do modelo e preparação dos dados
2. `train.py` - Script de treinamento e avaliação
3. `predict.py` - Script para fazer predições em novas imagens
4. `web/index.html` - Página web única contendo informações do modelo e interface para teste da IA através da API
5. `web/script.js` - Código javascript para funcionamento da página e integração com a API de predição do modelo
6. `api.py` - API em flask para testar a predição do modelo

## Arquitetura do Modelo

O modelo utiliza uma abordagem de **transfer learning** com MobileNetV2 como base:

### Configurações Principais
- **Tamanho da imagem**: 224x224 pixels (padrão para MobileNetV2)
- **Batch size**: 32
- **Taxa de aprendizado inicial**: 1e-4 com decaimento exponencial
- **Épocas**: 50 (com early stopping)

### Estrutura do Modelo
1. **Camada Base**: MobileNetV2 pré-treinada no ImageNet
   - Congelamento de todas as camadas exceto as últimas 20
   - Pooling médio global para redução dimensional

2. **Camadas Adicionais**:
   - Dropout (20%) para regularização
   - Batch Normalization para estabilização
   - Camada Dense final com ativação softmax

### Otimização
- **Otimizador**: Adam com weight decay (1e-4)
- **Função de perda**: Categorical Crossentropy com label smoothing (0.1)
- **Métricas**: Acurácia

## Treinamento

### Preparação dos Dados
- **Data Augmentation** (apenas para treino):
  - Rotação (15°)
  - Deslocamentos (15%)
  - Zoom (15%)
  - Inversão horizontal
  - Ajustes de brilho (10%)
  - Normalização (valores entre 0-1)

- **Geração de pesos** para classes desbalanceadas

### Processo de Treinamento
1. **Callbacks**:
   - Early Stopping (paciência=42 épocas)
   - Redução de LR no platô (fator=0.5, paciência=5)
   - Checkpoint do melhor modelo

2. **Avaliação**:
   - Acurácia, Precisão, Recall, F1-Score
   - ROC-AUC para multiclasse
   - Matriz de confusão
   - Gráficos de evolução do treino