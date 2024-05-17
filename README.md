# COMP6258_Reproducibility_Challenge
COMP6258 Differntiable Programming and Deep Learning Coursework

In this report, we are reimplementing the deep reinforcement learning experiments from ["Deep Reinforcement Learning for Cost-Effective Medical Diagnosis"](https://openreview.net/forum?id=0WVNuEnqVu) to predict Acute Kidney Injury diagnosis. This report will give a breakdown of different experiments carried out and a comparison of how well they perform to the results obtained in the paper. This report will then conclude with an evaluation if the experiments are reproducible, and assess if our results support the conclusions drawn from the paper.

# Data Processing
- path_to_new_file.csv: raw data
- data_cleaning.ipynb: data cleaner
- cleaned_data.csv: Data after cleaning by using the data_cleaning.ipynb

First, we extract the variables needed in the paper from the original data set, such as 'lactate\_max', 'ph\_min', 'ph\_max', 'so2\_min', etc. Then we dealt with missing values and outliers.To maintain integrity of the data, we deleted all columns and rows with more than 5\% missing data. After initial cleaning, the data set remained with 1786 observations and 47 variables.

For categorical data, we convert it into an int type for model identification. For example, in patient ethnicity, 0 represents "unknown", 1 represents "white", 2 represents "minorities", and 3 represents "black\_african\_american"

We standardized the numerical data to eliminate the differences between different feature dimensions and make model training more stable. For specific data processing content, you can view data\_cleaning.ipynb in the project GitHub.\\
Finally, before training each model, we divided the data set into a training set (70\%), a validation set (15\%), and a test set (15\%) for model training, hyperparameter tuning, and final evaluation.

# Baseline Model
- LR&rRF.ipynb: Logist regression and random forest
- DNN.ipynb: three-layers DNN
- XGBoost.ipynb: XGBoost

## DNN
1. The first hidden layer is a fully connected layer (fc1), which can map the input data to a 128-dimensional hidden layer. Then the batch normalization layer (bn1) performs batch normalization on the output of the first hidden layer, normalizing the output of each batch to speed up training. Then use ReLU as the activation function and randomly drop some neurons with 50\% probability in the Dropout layer (dropout) to prevent overfitting.

2. The second hidden layer is the same as the first. First, the fully connected layer (fc2) and then the batch normalization layer (bn2) also use ReLU as the activation function and have 50\% Dropout.

3. Finally is the output layer. This is also a fully connected layer (fc3) which maps the output of the second hidden layer to a 3-dimensional output layer, representing predictions for three categories.

## XGBoost
1. For our implementation, once the data has been split into the training and test set, and feature scaling has been performed, we made a decision tree and fit the data to this model. This is done by splitting the data based on the features. To find the best split, we calculated the gain of each split based on the gradient and Hessians and chose the optimum one. The tree can then be built recursively and predict values for the training set. 

2. The XGBoost classifier implements the gradient boosting algorithm using the DecisionTree class as the weak learner. It trains the model by iteratively adding decision trees to fit the residuals of the previous trees. Then, predictions can be made by traversing through the final concatenated tree

# SM-DDPO
The end-to-end semi-model-based RL training framework illustrated below contains three core modules: Posterior State Encoder via Imputer (E), Classifier (C), and Panel/Predictor Selector (S). The final state embedding of RL contains the observation 0-1 indicator, embedding output by (E) and (C).
<p align="center">
<img width="668" alt="image" src="https://user-images.githubusercontent.com/41489420/221870344-4b573367-0801-47f3-a644-f537f7d78271.png">
</p>

- classifier.py: (C) module
- flow_models.py, nflow.py, imputation.py: (E) module, where a flow-based deep imputer named [EMFlow](https://github.com/guipenaufv/EMFlow) is used.
- rl.py: (S) module
- SM-DDPO.ipynb: In order to comply with the data used in this reproducibility report, we modified its model to a certain extent.
  




