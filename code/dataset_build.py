import os
import cv2
import random
from torch.utils.data import Dataset


def get_dataset(dataset_path, train_tresh, train_transform=None, test_transform=None, target_transform=None, seed=993):
        
        # define train and test datasets
        train_dataset = SignDataset(train_transform, target_transform)
        test_dataset = SignDataset(test_transform, target_transform)
        
        # train tresh to divide the original dataset
        if not (train_tresh <= 1 and train_tresh >= 0):
            # MISSING ERROR
            return
        
        # take the name of the directories 
        dir_list = sorted(os.listdir(dataset_path))
        
        rng = random.Random(seed)
        
        class_id = 0
        
        # iterate over the directories (each directory has the name that corresponds with the label)
        for symbol in dir_list:
            
            #add the single directory to the main path
            symbol_path = os.path.join(dataset_path, symbol)
            
            # there exists a readme.md file, skip it
            try:
                img_list = os.listdir(symbol_path)
            except NotADirectoryError as e:
                continue
            
            # randomize the pick of paths, so the split train-test is consistent with the LIS dataset
            img_list = img_list.copy()
            rng.shuffle(img_list)
            
            # number of images per directoty
            dim = len(img_list)
            
            # number of images in the train set
            idx_thres = int(dim * train_tresh)
            
            # iterate over the images
            for i, image in enumerate(img_list):
                
                # add the image path to the main one
                img_path = os.path.join(symbol_path, image)
                
                # split train and test based on the index
                # CHECK: randomize this pick
                if i < idx_thres:
                    train_dataset.add_example(img_path, symbol, class_id)
                else:
                    test_dataset.add_example(img_path, symbol, class_id)
                    
            class_id += 1
        
        return train_dataset, test_dataset
        

class SignDataset(Dataset):
    
    # added transform and target_transform to init
    def __init__(self, transform=None, target_transform=None):
        self.dataset = {}
        self.label_to_idx = {}
        self.idx_to_label = {}
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.dataset)
    
    # add example as image path, label
    def add_example(self, path, label, label_idx):
        id = len(self.dataset)
        self.dataset[id] = (path, label)
        self.label_to_idx[label] = label_idx
        self.idx_to_label[label_idx] = label
        
    def __str__(self):
        string = ""
        for id in range(len(self.dataset)):
            string += f"| id: {id}, \t{self.dataset[id][0]}, \t\t{self.dataset[id][1]} ({self.label_to_idx[self.dataset[id][1]]})\n"
        return string
    
    def __getitem__(self, idx):
        img_path = self.dataset[idx][0]
        label = self.dataset[idx][1]
        label_idx = self.label_to_idx[label]
        
        # upload the image
        img = cv2.imread(img_path)
        
        # transform the image from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label_idx)   
        
        return img, label
    
    def num_classes(self):
        return len(self.idx_to_label)
    
    def get_classes(self):
        classes = []
        for i in range(self.num_classes()):
            classes.append(self.idx_to_label[i])
            
        return classes
