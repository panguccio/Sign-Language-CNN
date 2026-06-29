import os
import cv2
from torchvision.io import decode_image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


path = "/Users/anna/.cache/kagglehub/datasets/nicholasnisopoli/lisdataset/versions/1/LIS-fingerspelling-dataset"

def get_dataset(dataset_path, train_tresh, image_size=(64, 64)):
        
        # crea i due dataset vuoti
        train_dataset = SignDataset(image_size)
        test_dataset = SignDataset(image_size)
        
        # check se il treshold sia tra 0 e 1
        if not (train_tresh <= 1 and train_tresh >= 0):
            return # add error
        
        # prende l'elenco di cartelle dentro il path del dataset
        dir_list = sorted(os.listdir(dataset_path))
        
        class_id = 0
        # suppone che ogni cartella sia associata a un simbolo
        
        for symbol in dir_list:
            
            # unisce i 2 path
            symbol_path = os.path.join(dataset_path, symbol)
            
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
                
                if i < idx_thres: # lo mette dentro il train, altrimenti test
                    train_dataset.add_example(img_path, symbol, class_id)
                else:
                    test_dataset.add_example(img_path, symbol, class_id)
                    
            class_id += 1
        
        return train_dataset, test_dataset
        

class SignDataset(Dataset):
    
    # inizializza il dataset
    def __init__(self, image_size):
        self.dataset = {}
        self.image_size = image_size
        self.label_to_idx = {}
        
    def __len__(self):
        return len(self.dataset)
    
    # add example as image path, label and index
    def add_example(self, path, label, label_idx):
        id = len(self.dataset)
        self.dataset[id] = (path, label)
        self.label_to_idx[label] = label_idx
        
    def __str__(self):
        string = ""
        for id in range(len(self.dataset)):
            string += f"| id: {id}, \t{self.dataset[id][0]}, \t\t{self.dataset[id][1]} ({self.label_to_idx[self.dataset[id][1]]})\n"
        return string
    
    def __getitem__(self, idx):
        img_path = self.dataset[idx][0]
        label = self.dataset[idx][1]
        label_idx = self.label_to_idx[label]
        
        # carica l'immagine
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        
        transform = transforms.ToTensor()
        img_tensor = transform(img)
        label_tensor = torch.tensor(label_idx)
        
        return (img_tensor, label_tensor)
    
    def num_classes(self):
        return len(self.label_to_idx)
