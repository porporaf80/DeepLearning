# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented. 
# If you run this notebook on a different environment, e.g. your desktop, you may need to uncomment and install certain libraries.

#!pip install numpy==1.21.4
#!pip install pandas==1.3.4
#!pip install keras==2.1.6

import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)

concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()
#print (concrete_data)
print(concrete_data.shape)

#verifica dei dati nulli
concrete_data.describe()

concrete_data.isnull().sum()

#Dividiamo il data set in predictors e Target
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#normalizziamo i predictors 

predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()

#prendiamo il numero di predittori perchè ci serve per definire la rete neurale

n_cols = predictors_norm.shape[1] # number of predictors

import keras
from keras.models import Sequential
from keras.layers import Dense