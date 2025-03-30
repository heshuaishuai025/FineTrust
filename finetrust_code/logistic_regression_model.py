import time
import argparse
import numpy as np
from texttable import Texttable
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression,LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from math import sqrt

def tab_printer(args):

    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def calculate_auc(prediction, test_label):
    # auc = roc_auc_score(test_label, predcition_score)

    f1 = f1_score(test_label, prediction)
    f1_micro = f1_score(test_label, prediction, average="micro")
    f1_macro = f1_score(test_label, prediction, average="macro")
    f1_weighted =f1_score(test_label, prediction, average="weighted")
     

    mae_convert = {0:-1, 1:-0.9, 2:-0.8, 3:-0.7, 4:-0.6, 5:-0.5, 6:-0.4, 7:-0.3, 8:-0.2, 9:-0.1, 10:0.1, 11:0.2, 12:0.3, 13:0.4, 14:0.5, 15:0.6, 16:0.7, 17:0.8, 18:0.9, 19:1}
    label_mae = [mae_convert[a] for a in test_label]
    prediction_mae = [mae_convert[a] for a in prediction]

    mae = mean_absolute_error(label_mae, prediction_mae)
    rmse = sqrt(mean_squared_error(label_mae, prediction_mae))


    return f1, f1_micro, f1_macro, f1_weighted, mae, rmse

def best_printer(log):
    t = Texttable()
    t.set_precision(6)
    t.add_rows([per for per in log])
    print(t.draw())

def parameter_parser():

    parser = argparse.ArgumentParser(description = "Run TrustGAT.")

    parser.add_argument("--embedding-path",
                        nargs = "?",
                        default = "./data/results/embedding.txt",
                        help = "Target embedding txt.")

    parser.add_argument("--train-index",
                        nargs = "?",
                        default = "./data/results/train_index.txt")
    parser.add_argument("--train-label",
                        nargs = "?",
                        default = "./data/results/train_label.txt")

    parser.add_argument("--test-index",
                        nargs = "?",
                        default = ".data/results/test_index.txt")

    parser.add_argument("--test-label",
                        nargs = "?",
                        default = "./data/results/test_label.txt")


    return parser.parse_args()

def read_graph(args):

    embedding = []
    with open(args.embedding_path) as e:
        for emb in e:
            embedding.append(list(map(float, emb.split())))

    train = []
    with open(args.train_index) as o:
        for line in o:
            a = list(map(int, line.split()))[0]
            l = list(map(int, line.split()))[1]
            train.append(embedding[a] + embedding[l])

    # Directly insert values into train_label
    train_label = []
    with open(args.train_label) as x:
        train_label = [float(line.split()[0]) for line in x]

    test = []
    with open(args.test_index) as i:
        for line in i:
            a = list(map(int, line.split()))[0]
            l = list(map(int, line.split()))[1]
            test.append(embedding[a] + embedding[l])

    test_label = []
    with open(args.test_label) as te:
        test_label = [float(line.split()[0]) for line in te]

    return train, train_label, test, test_label


def linear_regression():

    beginning = time.time()
    args = parameter_parser()
    tab_printer(args)
    train, train_label, test, test_label = read_graph(args)

    # clf = LogisticRegression()
    clf = LinearRegression()
    clf.fit(train, train_label)

    # prediction_score = clf.predict_proba(test)[:, 1]
    prediction = clf.predict(test)
    # f1, f1_micro, f1_macro, f1_weighted, mae, rmse = calculate_auc(prediction, test_label)
     
    mae = mean_absolute_error(test_label, prediction)
    rmse = sqrt(mean_squared_error(test_label, prediction))
    # 创建一个包含结果的列表
    # results = [["F1", "F1_MICRO","F1-MACRO", "F1-WEIGHTED","MAE", "RMSE", "Time"],  f1, f1_micro, f1_macro, f1_weighted, mae, rmse, time.time() - beginning]
    results = [["MAE", "RMSE", "Time"],  mae, rmse, time.time() - beginning]
    # 输出到txt文件
    output_file_path = "results.txt"
    with open(output_file_path, "w") as file:
        for result in results:
            if isinstance(result, list):
                file.write("\t".join(map(str, result)) + "\n")
            else:
                file.write(str(result) + "\n")

if __name__ == "__main__":
    linear_regression()