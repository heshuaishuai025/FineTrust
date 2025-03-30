import numpy as np
from gcn import GCNTrainer
from arg_parser import parameter_parser
import time
from utils import tab_printer, read_graph, score_printer, best_printer #save_logs
from metrics import Reg
def main():
    args = parameter_parser()
    tab_printer(args)
    edges = read_graph(args)
    best = [["Run","Epoch","AUC","F1_micro","F1_macro","F1_weighted","F1","MAE", "RMSE", "Run Time"]]

    for t in range(1):
        trainer = GCNTrainer(args, edges)
        trainer.setup_dataset()
        print("Ready, Go! Round =", str(t))
        trainer.create_and_train_model()

if __name__ == "__main__":
    all_train=[]
    all_test=[]
    global_start = time.time()
    for i in range(10):
        start = time.time()
        main()  # embedding training
        end1 = time.time()
        print(f"!!!!!!!!!!!Embedding training time: {end1 - start:.2f} seconds")

        start1 = time.time()
        train_eval,test_eval = Reg()  # regression prediction
        end2 = time.time()
        print(f"assessment time: {end2 - start1:.2f} seconds")



   