import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from kaggle.api.kaggle_api_extended import KaggleApi

# Inicialize a API
api = KaggleApi()
api.authenticate()

# Baixe o dataset
api.dataset_download_files('gpiosenka/100-sports-image-classification', path='./', unzip=True)