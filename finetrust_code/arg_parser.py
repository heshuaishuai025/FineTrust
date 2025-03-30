import argparse

def parameter_parser():   #参数解析器

    parser = argparse.ArgumentParser(description = "Run TrustGAT.")

    #dataset pgp/ advogato
    parser.add_argument("--edge-path",
                        nargs = "?",
                        default = "../data/Main_Graph/bitcoin_otc.txt", # pgp-rating_4_label.txt
	                    help = "Edge list txt.")

    parser.add_argument("--features-path",
                        nargs = "?",
                        default ="../data/embeddings/bitcoin_otc_embed_sdgnn_64.txt", # pgp_node_vec.txt
                        help = "Node feature txt.")
    
    parser.add_argument("--embedding-path",
                        nargs = "?",
                        default = "../data/results/Embeddings_bitcoin_otc.csv",
                        help = "Target embedding txt.")
    
   
    parser.add_argument("--test-size",
                        type = float,
                        default = 0.2,
                        help = "Test dataset size. Default is 0.2.")

    parser.add_argument("--seed",
                        type = int,
                        default = 42,
                        help = "Random seed for sklearn pre-training. Default is 42.")

    parser.add_argument("--layers",
                        nargs="+",
                        type=int,
                        help = "Layer dimensions separated by space. E.g. 32 32.")

    parser.add_argument("--learning_rate",
                        type = float,
                        default = 0.01,
                        help = "Learning rate. Default is 0.01.")

    parser.add_argument("--weight-decay",
                        type = float,
                        default = 10**-5,
                        help = "Learning rate. Default is 10^-5.")

    parser.add_argument("--epochs",
                        type = int,
                        default =500,
                        help = "Number of training epochs. Default is 100.")

    parser.add_argument("--normalize_embedding",
                        action = "store_false",
                        help = "Normalize embedding. Default is False.") # False better than true
    parser.add_argument("--save_dir",
                        type = str,
                        default = '..//result//exp')
    parser.add_argument("--both_degree",
                        action = "store_true",
                        help = "Include both degree when aggregating. Default is True.") # True better than false

    parser.add_argument("--stratified_split",
                        action = "store_true",
                        help = "Split the data based on number of data in each category. Default is True.")

    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')

    parser.add_argument("--lama",
                        type=float,
                        default=1,
	                help="Embedding regularization parameter. Default is 1.0.  1.2.3.4.5,0.5")
    parser.add_argument("--lamb",
                        type=float,
                        default=1,
	                help="Embedding regularization parameter. Default is 1.0.  1.2.3.4.5,0.5")
    parser.add_argument("--nolink-size",
                        type = int,
                        default = 0,
                        help = "Nolink edges/linked edges when computing loss. Default is 1") #0, 1, 2
    parser.add_argument("--class_imbalance",
                        action = "store_false",
                        help = "Deal with imbalance classess. Default is False")

    parser.add_argument("--spectral-features",
                        dest="spectral_features",
                        action="store_true")

    parser.add_argument("--general-features",
                        dest="spectral_features",
                        action="store_false")
    parser.set_defaults(spectral_features=True)
    
    parser.add_argument("--reduction-iterations",
                        type=int,
                        default=30,
	                help="Number of SVD iterations. Default is 30.")

    parser.add_argument("--reduction-dimensions",
                        type=int,
                        default=64,
	                help="Number of SVD feature extraction dimensions. Default is 64.")
    
    
    parser.set_defaults(normalize_embedding = False)

    parser.set_defaults(layers = [32,32,32])

    parser.set_defaults(both_degree = True)

    parser.set_defaults(stratified_split = True)

    parser.set_defaults(class_imbalance = False)

    return parser.parse_args()