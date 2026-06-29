import os
import cv2

path = "/Users/anna/.cache/kagglehub/datasets/nicholasnisopoli/lisdataset/versions/1/LIS-fingerspelling-dataset"

class Preprocessor:
    def __init__(self, dataset_path, image_size=(64, 64)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        # check image_size is a tuple
        # train_tresh da passare al costruttore + check che sia tra 0 e 1

    def preprocess(self, train_tresh):
        
        # crea i due dataset vuoti
        train_dataset = Dataset()
        test_dataset = Dataset()
        
        # check se il treshold sia tra 0 e 1
        if not (train_tresh <= 1 and train_tresh >= 0):
            return # add error
        
        # prende l'elenco di cartelle dentro il path del dataset
        dir_list = os.listdir(self.dataset_path)
        
        # suppone che ogni cartella sia associata a un simbolo
        for symbol in dir_list:
            print(symbol)
            
            # unisce i 2 path
            symbol_path = os.path.join(self.dataset_path, symbol)
            
            # prende l'insieme di immagini nella cartella (errore se non è una cartella)
            try:
                img_list = os.listdir(symbol_path)
            except NotADirectoryError as e:
                continue
            
            # numero di immagini contenute
            dim = len(img_list)
            # numero di immagini che andranno nel train set
            idx_thres = int(dim * train_tresh)
            
            # per ogni immagine, la aggiunge al train o al test
            for i, image in enumerate(img_list):
                
                img_path = os.path.join(symbol_path, image)
                self.resize(img_path)
                
                if i < idx_thres: # lo mette dentro il train, altrimenti test
                    train_dataset.add_example(img_path, str(symbol))
                else:
                    test_dataset.add_example(img_path, str(symbol))
        
        return train_dataset, test_dataset
    
    def resize(self, img_path):
        img1 = cv2.imread(img_path)
        img2 = cv2.resize(img1, (64, 64))
        cv2.imwrite(img_path, img2)
        
        
        

class Dataset:
    def __init__(self):
        self.dataset = {}
        
    def add_example(self, path, label):
        id = len(self.dataset)
        self.dataset[id] = Image(path, label)
        
    def __str__(self):
        string = ""
        for id in range(len(self.dataset)):
            string += "| id: " + str(id) + " " + self.dataset[id].path + " " + self.dataset[id].label + " |\n"
        return string
         
class Image:
    def __init__(self, path, label):
        self.path =  path
        self.label = label     
    
    def __str__(self):
        return str(self.path) + " " + str(self.label)
    
lis_prp = Preprocessor(path)
train, test = lis_prp.preprocess(0.5)