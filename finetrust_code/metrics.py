import sklearn, sys
from sklearn.metrics import explained_variance_score, r2_score, classification_report, mean_squared_log_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from numpy.random import normal
import h5py
import pickle
import pandas as pd
import numpy as np
from MLP import *
import logging
import tensorflow as tf
import time



def predictMetricsRegression(embeddings, df_train, df_test):
    train = np.zeros((df_train.shape[0], 2*embeddings.shape[1]))
    y_train = np.zeros((df_train.shape[0],1))
    test = np.zeros((df_test.shape[0], 2*embeddings.shape[1]))
    y_test = np.zeros((df_test.shape[0],1))

    y_test = df_test['weight'].values
    y_train = df_train['weight'].values

    for i in range(0, df_train.shape[0]):
        train[i,:] = np.append(embeddings[int(df_train['source'].iloc[i])], embeddings[int(df_train['target'].iloc[i])])

    for i in range(0, df_test.shape[0]):
        test[i,:] = np.append(embeddings[int(df_test['source'].iloc[i])], embeddings[int(df_test['target'].iloc[i])])

    model = MLPRegressor(input_dim=int(2*embeddings.shape[1]), output_dims=1)


    start = time.time()
    history = model.fit(x=train, y=y_train, batch_size=128, epochs=20, verbose=1)
    train_eval = model.evaluate(x=train, y=y_train)
    test_eval = model.evaluate(x=test, y=y_test)
    end = time.time()
    print(f"!!!!!!!!!!!!!!!assessing time: {end - start:.2f} seconds")

    print("Evaluation Scores for Training Data ")
    print(train_eval)

    print("Evaluation Scores for Testing Data ")
    print(test_eval)
    return train_eval,test_eval


def Reg():
    file_path = '../data/results/Embeddings_bitcoin_otc.csv'
    feature = []
    with open(file_path) as vec:
        for node in vec:
            feature.append(node.split()[0:])  ########
    embeddings_otc = np.array(feature, np.float32)

    df_train_otc = pd.read_csv("../data/datasets/mod/train_otc.csv")
    df_test_otc = pd.read_csv("../data/datasets//mod/test_otc.csv")


    print("//n")
    print("Weight Prediction Scores for bitcoin OTC dataset")
    train_eval,test_eval= predictMetricsRegression(embeddings_otc, df_train_otc, df_test_otc)

      
    return  train_eval,test_eval  


if __name__ == "__main__":
    Reg()
