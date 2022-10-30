"""Functions for preprocessing of input data"""
from helpers import *
import numpy as np


def preprocessing(input_x, input_y, pca_num=False, outlier=False, degree=4):
    """
    Main preprocessing function.
    Arguments: input_x: raw feature matrix
               input_y: raw label vector
               pca_num: the dimension of output feature matrix after PCA. False if PCA is not included.
               outlier: the cleaning parameter for outlier removal (see in the report). False if outlier removal is not included.
               degree: the polynomial degree (see in the report).
    """
    output_x = input_x.copy()
    output_x = replace_999_feature1(output_x)
    if outlier:
        output_x, input_y = remove_outlier(output_x, input_y, outlier)
    output_x = one_hot(output_x)
    if pca_num:
        output_x1 = output_x[:, 4:]
        output_x1pca, _ = pca(output_x1, pca_num)
        output_x = np.concatenate([output_x[:, :4], output_x1pca], axis=1)

    f = output_x.shape[1]
    output_x = nonlinear(output_x, degree + 1)

    for i in range(4):
        for j in range(4, output_x.shape[1], 5):
            output_x = interactions(output_x, i, j)

    for i in range(4, f - 1, 1):
        for j in range(i + 1, f, 1):
            output_x = interactions(output_x, i, j)

    return output_x, input_y


def replace_999_feature1(input_x, method='median'):
    """
    Replace the NULL values in the first feature vector.
    Arguments: input_x: feature matrix.
               method: choose between 'median' or 'mean' to determine the imputation method.
    """
    if method == 'median':
        input_x[np.where(input_x[:, 0] == -999), 0] = np.median(input_x[np.where(input_x[:, 0] != -999)[0], 0])
        return input_x
    elif method == 'mean':
        input_x[np.where(input_x[:, 0] == -999), 0] = np.mean(input_x[np.where(input_x[:, 0] != -999)[0], 0])
        return input_x
    else:
        raise ValueError("Method not supported")


def replace_999_feature_other(input_x):
    """
    Replace the NULL values in the feature vectors except that ones in the first vector.
    Arguments: input_x: feature matrix.
    """
    input_x[input_x == -999] = np.nan
    input_x = standardize_data(input_x)
    input_x[np.isnan(input_x)] = 0
    return input_x


def one_hot(input_x):
    """
    One-hot encoding and missing value imputation.
    Arguments: input_x: feature matrix.
    """
    cat_col = input_x[:, 22]
    cat_mat = np.zeros((len(cat_col), 4))

    cat_mat[cat_col == 0, 0] = 1
    cat_mat[cat_col == 1, 0] = 1
    cat_mat[cat_col == 2, 0] = 1
    cat_mat[cat_col == 3, 0] = 1

    input_x = np.delete(input_x, 22, axis=1)
    output_x = replace_999_feature_other(input_x)

    return np.hstack((cat_mat, output_x))


def remove_outlier(input_x, input_y, k):
    """
    Remove outliers in the data matrix.
    Arguments: input_x: feature matrix.
               input_y: label vector.
               k: outlier removal parameter (see in the report).
    """
    input_x[input_x == -999] = np.nan
    mean_std_dict = get_mean_std(input_x)
    for key, value in mean_std_dict.items():
        option1 = list((value[0] - k * value[1]) < input_x[:, key])
        option2 = list(input_x[:, key] < (value[0] + k * value[1]))
        option3 = list(np.isnan(input_x[:, key]))
        option4 = [i and j for i, j in zip(option1, option2)]
        option5 = [i or j for i, j in zip(option3, option4)]
        input_x = input_x[option5]
        input_y = input_y[option5]
    input_x[np.isnan(input_x)] = -999
    return input_x, input_y


def nonlinear(input_x, degree):
    """
    Use polynomial feature expansion.
    Arguments: input_x: feature matrix.
               degree: polynomial degree (see in the report).
    """
    n, f = input_x.shape
    output_x = np.zeros((n, (degree * (f - 4)) + 4))
    output_x[:, :4] = input_x[:, :4]
    for feature in range(f - 4):
        for i in range(1, degree + 1):
            output_x[:, 3 + feature * degree + i] = (input_x[:, 4 + feature]) ** i

    return output_x


def interactions(x, i, j):
    """
    Use feature interaction to create new cross-term features.
    Arguments: x: feature matrix.
               i: index of feature vector 1 for interaction.
               j: index of feature vector 2 for interaction.
    """
    x_new = x[:, i] * x[:, j]
    x_new = x_new.reshape(len(x_new), 1)
    x_augmented = np.hstack((x, x_new))

    return x_augmented


