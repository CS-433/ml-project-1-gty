# -*- coding: utf-8 -*-
"""Main function for training and testing."""

from helpers import *
from preprocessing import preprocessing
from implementations import *

# Load the training and testing data
train_data_path = 'D:\\ML_Course\\Project1\\Final submission\\data\\train.csv'
test_data_path = 'D:\\ML_Course\\Project1\\Final submission\\data\\test.csv'
y_train, X_train, id_train = load_data(train_data_path)
y_test, X_test, id_test = load_data(test_data_path)

# Preprocess the raw data
tx_train, y_train = preprocessing(X_train, y_train, outlier=4, degree=4)
tx_test, y_test = preprocessing(X_test, y_test, degree=4)


# K-fold cross validation to find the best hyper parameter and visualization
lambdas, avg_err_tr, avg_err_te, best_lambda, best_rmse, best_w = cross_validation_ridge_demo(y_train, tx_train, 5, np.logspace(-3, 3, 7))
cross_validation_visualization(lambdas, avg_err_tr, avg_err_te)


# Use the best weights for prediction
y_pred = regression_prediction(tx_test, best_w)
y_pred[y_pred == 0] = -1
create_csv_submission(id_test, y_pred, "submission.csv")
