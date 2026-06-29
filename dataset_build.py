import os

path = "/Users/anna/.cache/kagglehub/datasets/nicholasnisopoli/lisdataset/versions/1/LIS-fingerspelling-dataset"

class Preprocessor:
    def __init__(self, dataset_path, image_size=(64, 64)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        # train_tresh da passare al costruttore + check che sia tra 0 e 1

    def preprocess(self):
        dir_list = os.listdir(self.dataset_path)
        for symbol in dir_list:
            symbol_path = os.path.join(self.dataset_path, symbol)
            try:
                img_list = os.listdir(symbol_path)
            except NotADirectoryError as e:
                continue
            dim = len(img_list)
            # idx_thres = int(dim * train_tresh)
            for i, image in enumerate(img_list):
                img_path = os.path.join(symbol_path, image)
                # if i < idx_thres lo mette dentro il train, altrimenti test
                
            # idea: creare 2 dataset: uno per train e uno per test
            # o 1 dataset con label train / test
                
        pass
    

class Dataset:
    # def __init__
    
    
    
# lis_prp = Preprocessor(path)
# lis_prp.preprocess()