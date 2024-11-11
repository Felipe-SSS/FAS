from fas_models import getModel_Regression
from fas_models import getModel_Classification
import mat73
import os
import sys
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tf_cfc import CfcCell
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Insert the found architectures here
# ============================================================================================================================================================ #
reg_params_5 = {'activation': 'tanh', 'cnn': 0, 'filters': 454, 'kernel_size': 3, 'lr': 0.03167968134479988, 'dense': 1, 'lstm': 2, 'pool': 'max', 'optimizer': 'adam', 'scaler': 'minmax', 'pca': 'yes', 'cells_0': 86, 'cells_1': 7, 'nodes_0': 113, 'dropout_0': 0.5}
reg_params_6 = {'activation': 'linear', 'cnn': 1, 'filters': 461, 'kernel_size': 1, 'lr': 6.710011371657373e-05, 'dense': 0, 'lstm': 2, 'pool': 'avg', 'optimizer': 'adam', 'scaler': 'std', 'pca': 'no', 'cells_0': 37, 'cells_1': 45}
reg_params_7 = {'activation': 'tanh', 'cnn': 2, 'filters': 77, 'kernel_size': 1, 'lr': 0.0020414364461062645, 'dense': 1, 'lstm': 2, 'pool': 'max', 'optimizer': 'nadam', 'scaler': 'minmax', 'pca': 'yes', 'cells_0': 70, 'cells_1': 88, 'nodes_0': 104, 'dropout_0': 0.5}
reg_params_8 = {'activation': 'sigmoid', 'cnn': 0, 'filters': 175, 'kernel_size': 5, 'lr': 0.012126574939241838, 'dense': 1, 'lstm': 1, 'pool': 'avg', 'optimizer': 'adam', 'scaler': 'minmax', 'pca': 'no', 'cells_0': 9, 'nodes_0': 34, 'dropout_0': 0.5}
reg_params_9 = {'activation': 'sigmoid', 'cnn': 0, 'filters': 59, 'kernel_size': 5, 'lr': 0.0026409392600291277, 'dense': 1, 'lstm': 1, 'pool': 'avg', 'optimizer': 'adam', 'scaler': 'std', 'pca': 'no', 'cells_0': 73, 'nodes_0': 251, 'dropout_0': 0.4}
reg_params_10 = {'activation': 'tanh', 'cnn': 0, 'filters': 289, 'kernel_size': 3, 'lr': 0.0008275765895813103, 'dense': 3, 'lstm': 2, 'pool': 'max', 'optimizer': 'adam', 'scaler': 'minmax', 'pca': 'no', 'cells_0': 53, 'cells_1': 70, 'nodes_0': 244, 'dropout_0': 0.0, 'nodes_1': 258, 'dropout_1': 0.0, 'nodes_2': 138, 'dropout_2': 0.5}
reg_params_15 = {'activation': 'linear', 'cnn': 2, 'filters': 65, 'kernel_size': 4, 'lr': 0.005328100254873861, 'dense': 0, 'lstm': 1, 'pool': 'avg', 'optimizer': 'adam', 'scaler': 'std', 'pca': 'no', 'cells_0': 95}

reg_params = {5: reg_params_5, 6: reg_params_6, 7: reg_params_7, 8: reg_params_8, 9: reg_params_9, 10: reg_params_10, 15: reg_params_15}
# ============================================================================================================================================================ #
class_params_5 = {'activation': 'relu', 'cnn': 1, 'filters': 326, 'kernel_size': 2, 'lr': 1.3575139713492116e-05, 'dense': 0, 'lstm': 1, 'pool': 'avg', 'optimizer': 'adam', 'scaler': 'std', 'pca': 'yes', 'cells_0': 83, 'loss': 'binary'}
class_params_6 = {'activation': 'tanh', 'cnn': 2, 'filters': 36, 'kernel_size': 4, 'lr': 0.0008523979693627619, 'dense': 1, 'lstm': 1, 'pool': 'max', 'optimizer': 'adam', 'scaler': 'none', 'pca': 'no', 'cells_0': 11, 'nodes_0': 289, 'dropout_0': 0.4, 'loss': 'f1'}
class_params_7 = {'activation': 'tanh', 'cnn': 0, 'filters': 126, 'kernel_size': 4, 'lr': 5.129276091104445e-05, 'dense': 1, 'lstm': 1, 'pool': 'max', 'optimizer': 'adam', 'scaler': 'none', 'pca': 'no', 'cells_0': 78, 'nodes_0': 226, 'dropout_0': 0.0, 'loss': 'binary'}
class_params_8 = {'activation': 'linear', 'cnn': 1, 'filters': 374, 'kernel_size': 5, 'lr': 0.0004370192870229064, 'dense': 1, 'lstm': 1, 'pool': 'max', 'optimizer': 'nadam', 'scaler': 'std', 'pca': 'no', 'cells_0': 64, 'nodes_0': 291, 'dropout_0': 0.5, 'loss': 'f1'}
class_params_9 = {'activation': 'relu', 'cnn': 2, 'filters': 500, 'kernel_size': 1, 'lr': 0.00014469136011607166, 'dense': 2, 'lstm': 1, 'pool': 'max', 'optimizer': 'nadam', 'scaler': 'none', 'pca': 'no', 'cells_0': 51, 'nodes_0': 40, 'dropout_0': 0.5, 'nodes_1': 193, 'dropout_1': 0.5, 'loss': 'f1'}
class_params_10 = {'activation': 'sigmoid', 'cnn': 0, 'filters': 486, 'kernel_size': 5, 'lr': 8.405541453221293e-05, 'dense': 2, 'lstm': 1, 'pool': 'avg', 'optimizer': 'adam', 'scaler': 'std', 'pca': 'yes', 'cells_0': 31, 'nodes_0': 242, 'dropout_0': 0.2, 'nodes_1': 195, 'dropout_1': 0.1, 'loss': 'f1'}
class_params_15 = {'activation': 'linear', 'cnn': 2, 'filters': 349, 'kernel_size': 1, 'lr': 0.00099673616465655, 'dense': 0, 'lstm': 2, 'pool': 'avg', 'optimizer': 'nadam', 'scaler': 'std', 'pca': 'no', 'cells_0': 59, 'cells_1': 42, 'loss': 'binary'}

class_params = {5: class_params_5, 6: class_params_6, 7: class_params_7, 8: class_params_8, 9: class_params_9, 10: class_params_10, 15: class_params_15}
# ============================================================================================================================================================ #

def get_flops(model):
    # Save the current file descriptor for stdout
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)

    # Create a temporary file to redirect stdout
    with open(os.devnull, 'w') as devnull:
        os.dup2(devnull.fileno(), stdout_fd)
        
        # Execute profiling
        forward_pass = tf.function(model.call, input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:])])
        graph_info = profile(forward_pass.get_concrete_function().graph, options=ProfileOptionBuilder.float_operation())
        flops = graph_info.total_float_ops

    # Restore the original stdout file descriptor so that printing to stdout works normally again
    os.dup2(saved_stdout_fd, stdout_fd)
    os.close(saved_stdout_fd)

    return flops


for i in range(5, 11):

    model = getModel_Regression(reg_params[i], i)

    flops = get_flops(model)
    params = model.count_params()
    
    print(f"From the regression models using {i} observed ports: ")
    print('Flops: ', flops)
    print('Params: ', params)
    print("========================================================")

model = getModel_Regression(reg_params[15], 15)

flops = get_flops(model)
params = model.count_params()
    
print("From the regression models using 15 observed ports: ")
print('Flops: ', flops)
print('Params: ', params)
print("========================================================")

del flops
del model
del params

for i in range(5, 11):

    model = getModel_Classification(class_params[i], i)

    flops = get_flops(model)
    params = model.count_params()
    
    print(f"From the classification models using {i} observed ports: ")
    print('Flops: ', flops)
    print('Params: ', params)
    print("========================================================")

model = getModel_Classification(class_params[15], 15)

flops = get_flops(model)
params = model.count_params()
    
print("From the classification models using 15 observed ports: ")
print('Flops: ', flops)
print('Params: ', params)
print("========================================================")

del flops
del model
del params