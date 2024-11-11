import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable all GPUS
  tf.config.set_visible_devices(physical_devices[0], 'GPU')
  visible_devices = tf.config.get_visible_devices()
  print('Visible devices:', visible_devices)
except:
  print('Invalid device or cannot modify virtual devices once initialized.')
  pass

# Import all the necessary libraries.
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy.random as rnd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import csv
from autogluon.tabular import TabularDataset, TabularPredictor
import autokeras as ak
from autokeras import StructuredDataClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import accuracy_score
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Input
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import Conv1D
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.backend import clear_session
import optuna
from sklearn.decomposition import PCA
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


def getModel_Regression(params, num_ports):    
    
    activation = params['activation']
    cnn = params['cnn']
    filters = params['filters']
    kernel_size = params['kernel_size']
    learning_rate = params['lr']
    dense = params['dense']
    lstm = params['lstm']
    
    pool = params['pool']
    if(pool == 'max'):
        pool_layer = MaxPooling1D(pool_size=1)
    else:
        pool_layer = AveragePooling1D(pool_size=1)

    optimizer = params['optimizer']
    if optimizer=='adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer=='nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer=='sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    scaler = params['scaler']
    if(scaler == 'std'):
        scl = StandardScaler()
    else:
        scl = MinMaxScaler(feature_range=(-1, 1))

    pca = params['pca']
    if(pca == 'yes'):
        pCA = PCA()
    
    # define model
    model = Sequential()
    if(cnn>0):
        sp=(2048, 1, num_ports)
    else:
        sp=(1, num_ports)
    model.add(Input(shape=sp))
    
    if(cnn>0):
        model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')))
        model.add(TimeDistributed(pool_layer))
        model.add(TimeDistributed(Flatten()))
    
    for i in range(lstm):
        cells = params['cells_'+str(i)]
        if(lstm > 1 and i < lstm-1):
            return_sequences=True
        else:
            return_sequences=False
        model.add(LSTM(cells, activation=activation, return_sequences=return_sequences))
    for i in range(dense):
        nodes = params['nodes_'+str(i)]
        model.add(Dense(nodes, activation='relu'))
        dropout = params['dropout_'+str(i)]
        model.add(Dropout(dropout))
    model.add(Dense(100, activation='linear'))

    return model

# ================================================================================ #
# ================================================================================ #
# ================================================================================ #

def getModel_Classification(params, num_ports):    
    
    activation = params['activation']
    cnn = params['cnn']
    filters = params['filters']
    kernel_size = params['kernel_size']
    learning_rate = params['lr']
    dense = params['dense']
    lstm = params['lstm']
    
    pool = params['pool']
    if(pool == 'max'):
        pool_layer = MaxPooling1D(pool_size=1)
    else:
        pool_layer = AveragePooling1D(pool_size=1)

    optimizer = params['optimizer']
    if optimizer=='adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer=='nadam':
        opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer=='sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    scaler = params['scaler']
    if(scaler == 'std'):
        scl = StandardScaler()
    else:
        scl = MinMaxScaler(feature_range=(-1, 1))

    pca = params['pca']
    if(pca == 'yes'):
        pCA = PCA()
    
    # define model
    model = Sequential()
    if(cnn>0):
        sp=(2048, 1, num_ports)
    else:
        sp=(1, num_ports)
    model.add(Input(shape=sp))
    
    if(cnn>0):
        model.add(TimeDistributed(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')))
        model.add(TimeDistributed(pool_layer))
        model.add(TimeDistributed(Flatten()))
    
    for i in range(lstm):
        cells = params['cells_'+str(i)]
        if(lstm > 1 and i < lstm-1):
            return_sequences=True
        else:
            return_sequences=False
        model.add(LSTM(cells, activation=activation, return_sequences=return_sequences))
    for i in range(dense):
        nodes = params['nodes_'+str(i)]
        model.add(Dense(nodes, activation='relu'))
        dropout = params['dropout_'+str(i)]
        model.add(Dropout(dropout))
    model.add(Dense(100, activation='sigmoid'))

    return model