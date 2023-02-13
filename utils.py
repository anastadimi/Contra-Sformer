import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from scipy import io
import numpy as np
import torch.nn.init as init
import torch.nn as nn
import os
import yaml
import argparse


# Device
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


# Task and Cross-Val Scheme
parser = argparse.ArgumentParser()
parser.add_argument("task", help="select a task", type=str, choices=['knot_tying','needle_passing','suturing'])
parser.add_argument("scheme", help="select a cross-validation scheme", type=str, choices=['loso','louo','4fold'])
args = parser.parse_args()
task = args.task
scheme = args.scheme

# Hyperparameters
with open('param.yaml') as f:
    param = yaml.load(f, Loader=yaml.FullLoader)

batch_size_train = param[task][scheme]['batch_size_train']
alpha = param[task][scheme]['alpha']
lr = param[task][scheme]['lr']
epochs = param[task][scheme]['epochs']

temp_padding = True
labels_norm = True
batch_size_val = 8 # 8 is the largest number across all folds in the dataset
ftrs = 512 # resnet18 features


# Task GRS statistics [std,mean]
task_stats = {'suturing': [5.400679719, 19.12820513], 'knot_tying': [5.106718264, 14.41666667], 'needle_passing': [4.821688216, 14.28571429]}


class spatial_features(Dataset):
    def __init__(self, task, labels_file, video_dir, temp_padding=None, labels_norm =None):
        self.task = task
        self.video_labels = pd.read_csv(labels_file,header=None)
        self.video_dir = video_dir
        self.temp_padding = temp_padding
        self.labels_norm = labels_norm
    
    def __len__(self):
        return len(self.video_labels)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 0])
        data = io.loadmat(video_path)
        video = data['F']
        video = torch.tensor(np.squeeze(video))
        label = torch.tensor(self.video_labels.iloc[idx, 1])
        if self.labels_norm: # Label normalization
            if self.task == 'suturing':
                label = (label - 19.12820513)/5.400679719
            elif self.task == 'knot_tying':
                label = (label - 14.41666667)/5.106718264 
            elif self.task == 'needle_passing':
                label = (label - 14.28571429)/4.821688216
        # Temporal padding: Padding the squences with zeros to make their length equal 
        # to the max length in order to use batch_size > 1
        if self.temp_padding: 
            if self.task == 'suturing':
                video = torch.cat((video,torch.zeros(1502-video.shape[0],ftrs)),0) 
            elif self.task == 'knot_tying':
                video = torch.cat((video,torch.zeros(629-video.shape[0],ftrs)),0) 
            elif self.task == 'needle_passing':
                video = torch.cat((video,torch.zeros(774-video.shape[0],ftrs)),0) 
        video = torch.permute(video,(1,0))

        return video, label


# Weight Initialization
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)

# Dataset
def dataset(task,scheme,fold,batch_size_train,batch_size_val,temp_padding):
    
    # Paths for videos and annotations (besed on the cross-validation schemes)  
    task = task
    scheme = scheme
    fold = fold
    cwd = os.getcwd()
    videos_dir = os.path.join(cwd,'resnet18_ftrs/' + task + '/' + scheme + '/' + fold +  'out')
    labels_train = os.path.join(cwd, 'splits/' + task + '/' + scheme + '/' + fold + 'out/train.csv')
    labels_val = os.path.join(cwd, 'splits/' + task + '/' + scheme + '/' + fold + 'out/val.csv')
    labels_ref = os.path.join(cwd, 'splits/' + task + '/ref.csv') # reference video
    
    # Data Splits
    train_set = spatial_features(task=task, labels_file=labels_train, video_dir=videos_dir, temp_padding=temp_padding, labels_norm=labels_norm)
    val_set = spatial_features(task=task, labels_file=labels_val, video_dir=videos_dir, temp_padding=temp_padding, labels_norm=labels_norm)
    ref_set = spatial_features(task=task, labels_file=labels_ref, video_dir=videos_dir, temp_padding=temp_padding, labels_norm=labels_norm)

    # Dataloaders
    train_loader = DataLoader(dataset=train_set,batch_size=batch_size_train,shuffle=True,pin_memory=True)
    val_loader = DataLoader(dataset=val_set,batch_size=batch_size_val,shuffle=False,pin_memory=True)
    ref_loader = DataLoader(dataset=ref_set,batch_size=1,shuffle=False,pin_memory=True)

    return train_loader, val_loader, ref_loader, train_set, val_set

# Unormalize labels
def unormalize(task,best_preds,labels_all_val):
    if labels_norm:
        best_preds = task_stats[task][0]*best_preds + task_stats[task][1]
        labels_all_val = task_stats[task][0]*labels_all_val + task_stats[task][1] 
    return best_preds, labels_all_val