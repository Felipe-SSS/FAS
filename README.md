This project gives support to reproduce the results presented in the paper **"Comparison of Classification and Regression Models for Port Selection in Fluid Antenna Systems"**. 

As described in Section **"III.E. Hyperparameter Optimization"**, we considered the Optuna framework to identify the optimal architecture for each scenario based on the number of observed ports. The tables below present architectures found in our analysis.

## Table: Architectures found for regression models for different numbers of observed ports
  
| Amount of Observed Ports | 5       | 6       | 7        | 8        | 9        | 10               | 15       |
|---------------------------|---------|---------|----------|----------|----------|------------------|----------|
| Activation Function       | Tanh    | Linear  | Tanh     | Sigmoid  | Sigmoid  | Tanh             | Linear   |
| CNN Layers                | 0       | 1       | 2        | 0        | 0        | 0                | 2        |
| CNN Filters               | 454     | 461     | 77       | 175      | 59       | 289              | 65       |
| DNN Layers                | 1       | 0       | 1        | 1        | 1        | 3                | 0        |
| DNN Nodes                 | 113     | -       | 104      | 34       | 251      | 244, 258, 138    | -        |
| DNN Dropout               | 0.5     | -       | 0.5      | 0.5      | 0.4      | 0.0, 0.0, 0.5    | -        |
| LSTM Layers               | 2       | 2       | 2        | 1        | 1        | 2                | 1        |
| LSTM Cells                | 86, 7   | 37, 45  | 70, 88   | 9        | 73       | 53, 70           | 95       |
| Pool Layer                | Max     | Avg     | Max      | Avg      | Avg      | Max              | Avg      |
| Kernel Size               | 3       | 1       | 1        | 5        | 5        | 3                | 4        |
| Learning Rate             | 0.0316  | 6.71e-5 | 0.00204  | 0.0121   | 0.00264  | 0.000827         | 0.00533  |
| Optimizer                 | Adam    | Adam    | NAdam    | NAdam    | Adam     | Adam             | Adam     |
| Scaler                    | MinMax  | Standard| MinMax   | MinMax   | Standard | MinMax           | Standard |
| PCA                       | Yes     | No      | Yes      | No       | No       | No               | No       |
| Loss Function             | MSE     | MSE     | MSE      | MSE      | MSE      | MSE              | MSE      |
| Total Parameters          | 46584   | 96619   | 117780   | 4488     | 68010    | 178464           | 74745    |
| FLOP Count                | 24395   | 13404394| 2562444  | 8833     | 135596   | 259612           | 16382005 |

## Table: rchitectures found for multi-label classification models for different numbers of observed ports

| Amount of Observed Ports | 5       | 6                | 7     | 8     | 9      | 10     | 15      |
|---------------------------|---------|------------------|-------|-------|--------|--------|---------|
| Activation Function       | Linear  | Linear           | Relu  | Relu  | Relu   | Tanh   | Sigmoid |
| CNN Layers                | 2       | 0                | 0     | 0     | 2      | 0      | 1       |
| CNN Filters               | 61      | 61               | 422   | 315   | 30     | 71     | 424     |
| DNN Layers                | 1       | 3                | 0     | 0     | 1      | 1      | 1       |
| DNN Nodes                 | 102     | 267, 283, 20     | -     | -     | 267    | 228    | 235     |
| DNN Dropout               | 0.1     | 0.1, 0.0, 0.5    | -     | -     | 0.1    | 0.3    | 0.4     |
| LSTM Layers               | 2       | 1                | 1     | 1     | 1      | 1      | 1       |
| LSTM Cells                | 65, 46  | 43               | 78    | 94    | 86     | 56     | 72      |
| Pool Layer                | Max     | Avg              | Avg   | Max   | Avg    | Avg    | Avg     |
| Kernel Size               | 4       | 1                | 1     | 1     | 1      | 5      | 5       |
| Learning Rate             | 2.7144e-05 | 7.3687e-05    | 0.0188| 6.4595e-04 | 5.0905e-04 | 1.0447e-04 | 1.6677e-04 |
| Optimizer                 | Nadam   | Adam             | Nadam | Adam  | Nadam  | Nadam  | Nadam   |
| Scaler                    | Standard| None             | Standard | Standard | MinMax | MinMax | MinMax |
| PCA                       | No      | No               | No    | Yes   | Yes    | No     | Yes     |
| Loss Function             | Binary  | Binary           | Binary| F1    | F1     | F1     | Binary  |
| Total Parameters          | 70003   | 103972           | 34732 | 48228 | 90577  | 50904  | 216115  |
| FLOP Count                | 5384107 | 207231           | 69286 | 96262 | 1408901| 71464  | 132356879 |

If you use any of these codes for research that results in publications, please cite our reference: [...]

