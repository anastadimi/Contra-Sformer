import torch
import statistics
from utils import *
from main import *
from models import *
import time



def cross_valid(fold, task, scheme):

    print(f'Training on fold {fold}')

    train_loader, val_loader, ref_loader, _, _ = dataset(task=task,scheme=scheme,fold=fold,
                                                    batch_size_train=batch_size_train,batch_size_val=batch_size_val,
                                                    temp_padding=temp_padding)

    x_ref, s_ref = next(iter(ref_loader)) # reference video (always the best)
    x_ref, s_ref = x_ref.to(device, dtype=torch.float32), s_ref.to(device, dtype=torch.float32)

    # Training
    _, best_coef, _, _, best_preds, labels_all_val = train(train_loader,val_loader,x_ref,s_ref,alpha)

    # Unormalize predictions and labels
    best_preds, labels_all_val = unormalize(task,best_preds,labels_all_val)

    return best_coef


# Define the folds for each cross-validation scheme
if scheme == 'loso':
    fold_list = ['1', '2', '3', '4', '5']
elif scheme == 'louo':
    fold_list = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
elif scheme == '4fold':
    fold_list = ['1', '2', '3', '4']
    
if scheme == 'louo' and task == 'needle_passing':
    fold_list = ['B', 'C', 'D', 'E', 'F', 'H', 'I'] # needle passing G trials are not provided in the JIGSAWS dataset

# Initialize list
folds_spearman = []
start = time.time()

# Training on each fold
for fold in fold_list:
    fold_best_coef = cross_valid(fold, task, scheme)
    folds_spearman.append(fold_best_coef)

end = time.time()
elapsed_time = end - start
print('>>> Training complete: {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))

# Print Average Spearman
print(f'Average Spearman: {statistics.mean(folds_spearman):.4f}')
