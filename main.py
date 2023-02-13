import copy, random
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from scipy import stats
from utils import *
from models import *

# -----------------------------------------------------------------------------

def train(train_loader, valid_loader, x_ref, s_ref, alpha):

    seed = 2
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Import model
    model = ContraSformer(spatial_ftrs_embed=ftrs, embed_dim=16, temp_window=25, dropout=0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, 600, gamma=0.1)
    
    # Loss function
    criterion_1 = nn.MSELoss().to(device)
    criterion_2 = nn.L1Loss().to(device)


    # -------------------------------------------------------------------------
    
    def iter_dataloader(data_loader,model,training):
    
        running_loss = 0.0
        labels_all = torch.empty((0)).to(device)
        output_all = torch.empty((0)).to(device)

        for iter, (inputs, labels) in enumerate(data_loader):
            
            # Move to device
            inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            # Clear grads
            if training == True:
                optimizer.zero_grad()
            
            # Forward pass
            s_pred = model.forward(inputs, x_ref, s_ref)

            # Loss  
            batch_loss = alpha*criterion_1(s_pred.squeeze(dim=1), labels) + (1-alpha)*criterion_2(s_pred.squeeze(dim=1), labels)
            
            if training == True:
                # Backprop
                batch_loss.backward()
                optimizer.step()
            
            # Running Loss
            running_loss += batch_loss.item()*inputs.size(0)
            
            # Used for metrics
            labels_all = torch.cat((labels_all, labels), 0)
            output_all = torch.cat((output_all, s_pred), 0)
        
        # Spearman Coefficient
        coef, _ = stats.spearmanr(labels_all.cpu().detach().numpy(),output_all.squeeze().cpu().detach().numpy())
        # Epoch Loss
        loss = running_loss / len(data_loader.dataset)

        return labels_all, output_all, coef, loss
    
    # -------------------------------------------------------------------------

    def training(model, train_loader):

        model.train()

        labels_all_train, output_all_train, coef_train, loss_epoch_train = iter_dataloader(train_loader, model, training=True)

        return labels_all_train, output_all_train, coef_train, loss_epoch_train

    # -------------------------------------------------------------------------

    def testing(model, valid_loader):

        model.eval()

        with torch.no_grad():

            labels_all_valid, output_all_valid, coef_valid, loss_epoch_valid = iter_dataloader(valid_loader, model, training=False)

        return labels_all_valid, output_all_valid, coef_valid, loss_epoch_valid

    # -------------------------------------------------------------------------

    # Initializations
    best_model_wts = copy.deepcopy(model.state_dict())
    best_coef = 0.0
    best_epoch = 0
    best_preds = torch.empty((0)).to(device)
    min_val_loss = np.inf

    # -------------------------------------------------------------------------

    for epoch in range(1, epochs + 1):

        training(model, train_loader)
        labels_all_val, output_all_val, coef_val, loss_epoch_val = testing(model, valid_loader)

        if loss_epoch_val < min_val_loss:
            min_val_loss = loss_epoch_val
            best_coef = coef_val
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_preds = output_all_val.squeeze()
        
        # Scheduler Update
        # scheduler.step()

    return best_model_wts, best_coef, min_val_loss, best_epoch, best_preds.detach(), labels_all_val.detach()
