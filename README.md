# neural-network-challenge-2

# Multi-Output Deep Learning Model for Employee Attrition and Department Prediction

This project demonstrates the implementation of a multi-output deep learning model using TensorFlow and Keras to predict employee attrition and department based on various employee attributes.

## Dataset

The dataset used in this project is the employee attrition dataset, which contains information about employees and their attributes such as age, department, distance from home, job satisfaction, and more. The dataset is loaded from a CSV file hosted on a remote server.

## Preprocessing

The preprocessing steps include:

1. Importing the necessary libraries and loading the dataset.
2. Creating separate DataFrames for the target variables (attrition and department) and the input features.
3. Splitting the data into training and testing sets.
4. Converting categorical variables to numeric using mapping and one-hot encoding.
5. Handling missing values and invalid values in the input features.
6. Scaling the input features using StandardScaler.

## Model Architecture

The multi-output deep learning model is created using the Functional API in Keras. The architecture includes:

1. An input layer that accepts the scaled input features.
2. Two shared dense layers with ReLU activation.
3. Separate branches for the department and attrition outputs, each consisting of a hidden layer and an output layer.
4. The department output layer uses softmax activation for multi-class classification, while the attrition output layer uses sigmoid activation for binary classification.

## Training and Evaluation

The model is compiled with the Adam optimizer, using categorical cross-entropy loss for the department output and binary cross-entropy loss for the attrition output. The model is trained for 100 epochs with a batch size of 32.

After training, the model is evaluated on the testing data, and the accuracy scores for both department and attrition predictions are printed.

## Summary

The project also includes a summary section that discusses the choice of accuracy as the evaluation metric, the selection of activation functions for the output layers, and potential ways to improve the model.

## Usage

To run the code, make sure you have the necessary dependencies installed, including TensorFlow, Keras, scikit-learn, pandas, and numpy. Execute the code cells in the provided order to preprocess the data, create the model, train it, and evaluate its performance.

Feel free to modify the code and experiment with different preprocessing techniques, model architectures, or hyperparameters to further improve the model's performance.
