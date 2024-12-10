This project gives support to reproduce the results presented in the paper **"Comparison of Classification and Regression Models for Port Selection in Fluid Antenna Systems"**. 

As described in Section **"III.E. Hyperparameter Optimization"**, we considered the Optuna framework to identify the optimal architecture for each scenario based on the number of observed ports. The tables below present architectures found in our analysis.

## Table: Architectures found for regression models for different numbers of observed ports
  
| Amount of Observed Ports | 5       | 6       | 7        | 8        | 9        | 10               | 15       |
|---------------------------|---------|---------|----------|----------|----------|------------------|----------|
| Activation Function       | Tanh    | Linear  | Tanh     | Sigmoid  | Sigmoid  | Tanh             | Linear   |
| CNN Layers                | 0       | 1       | 2        | 0        | 0        | 0                | 2        |
| CNN Filters               | -       | 461     | 77       | -        | -        | -                | 65       |
| DNN Layers                | 1       | 0       | 1        | 1        | 1        | 3                | 0        |
| DNN Nodes                 | 113     | -       | 104      | 34       | 251      | 244, 258, 138    | -        |
| DNN Dropout               | 0.5     | -       | 0.5      | 0.5      | 0.4      | 0.0, 0.0, 0.5    | -        |
| LSTM Layers               | 2       | 2       | 2        | 1        | 1        | 2                | 1        |
| LSTM Cells                | 86, 7   | 37, 45  | 70, 88   | 9        | 73       | 53, 70           | 95       |
| Pool Layer                | -       | Avg     | Max      | -        | -        | -                | Avg      |
| Kernel Size               | -       | 1       | 1        | -        | -        | -                | 4        |
| Learning Rate             | 0.0316  | 6.71e-5 | 0.00204  | 0.0121   | 0.00264  | 0.000827         | 0.00533  |
| Optimizer                 | Adam    | Adam    | NAdam    | NAdam    | Adam     | Adam             | Adam     |
| Scaler                    | MinMax  | Standard| MinMax   | MinMax   | Standard | MinMax           | Standard |
| PCA                       | Yes     | No      | Yes      | No       | No       | No               | No       |
| Loss Function             | MSE     | MSE     | MSE      | MSE      | MSE      | MSE              | MSE      |

## Table: Architectures found for multi-label classification models with 10 classes for different numbers of observed ports

| Amount of Observed Ports | 5              | 6                | 7         | 8        | 9        | 10        | 15      |
|---------------------------|---------------|------------------|-----------|----------|----------|-----------|---------|
| Activation Function       | Tanh          | ReLU             | Sigmoid   | Linear   | Linear   | ReLU      | ReLU    |
| CNN Layers                | 0             | 0                | 0         | 0        | 2        | 0         | 1       |
| CNN Filters               | -             | -                | -         | -        | 33       | 110       | 277     |
| DNN Layers                | 3             | 0                | 0         | 0        | 1        | 2         | 1       |
| DNN Nodes                 | 242, 167, 224 | -                | -         | -        | 100      | 215, 129  | 152     |
| DNN Dropout               | 0.2, 0.0, 0.3 | -                | -         | -        | 0.5      | 0.3, 0.5  | 0.0     |
| LSTM Layers               | 2             | 1                | 1         | 2        | 1        | 1         | 2       |
| LSTM Cells                | 48, 31        | 71               | 75        | 52, 60   | 85       | 49        | 52, 52  |
| Pool Layer                | -             | -                | -         | -        | Avg      | Max       | Avg     |
| Kernel Size               | -             | -                | -         | -        | 4        | 4         | 3       |
| Learning Rate             | 0.005012      | 0.0001353        | 0.00179   | 5.23e-05 | 0.00023  | 0.0002107 | 0.0002037 |
| Optimizer                 | Adam          | Adam             | Adam      | Adam     | Nadam    | Nadam     | Nadam   |
| Scaler                    | MinMax        | Standard         | Standard  | None     | Standard | MinMax    | MinMax  |
| PCA                       | Yes           | Yes              | No        | Yes      | Yes      | Yes       | Yes     |
| Loss Function             | Binary        | Binary           | Binary    | F1       | F1       | F1        | Binary  |

If you use any of these codes for research that results in publications, please cite our reference: [...]

