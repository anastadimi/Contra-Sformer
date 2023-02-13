import torch
import torch.nn as nn
import torch.nn.functional as F

# Model Definition
class ContraSformer(nn.Module):
    def __init__(self, spatial_ftrs_embed,  embed_dim, temp_window, dropout):
        super(ContraSformer, self).__init__()
        
        # Single Stage TCN
        self.SingleStageTCN = SingleStageTCN(spatial_ftrs_embed, embed_dim, temp_window, dropout)

        # Self-Attention Block
        self.multihead_attn = nn.MultiheadAttention(embed_dim = embed_dim, num_heads =  8, batch_first=True) #, dropout = 0.5)

        # Fully Connected layer
        self.fc= nn.Linear(embed_dim,1)


    def forward(self, x, x_ref, s_ref):

        # Single Stage TCN
        f = self.SingleStageTCN(x)
        f_ref = self.SingleStageTCN(x_ref)
        f_ref = f_ref.repeat(x.shape[0], 1, 1) # b,L,c = (batch_size,sequence_length,features)

        # Self-Attention block
        query = f # b,L,c
        key =  f_ref # b,L,c
        value = f_ref
        attn_output, _ = self.multihead_attn(query, key, value)

        # Skip Connection
        f_dif = attn_output + query

        # Scale dimensions
        f_dif = torch.permute(f_dif,(0,2,1)) # b,c,L
        s_ref = s_ref.repeat(x.shape[0], 1)

        # Temporal Average Pooling and Fully Connected Layer
        ds = self.fc(f_dif.mean(dim=2))
        
        # Add ds to s_ref to get the final prediction for the score
        s = ds + s_ref

        return s 
    
# Model Definition
class SingleStageTCN(nn.Module):
    def __init__(self, spatial_ftrs_embed, embed_dim, temp_window, dropout):
        super(SingleStageTCN, self).__init__()

        # conv layers
        self.convL1 = nn.Conv1d(spatial_ftrs_embed,64,temp_window,padding='same')
        self.convL2 = nn.Conv1d(64,32,temp_window,padding='same')
        self.convL3 = nn.Conv1d(32,16,temp_window,padding='same')
        self.convL4 = nn.Conv1d(16,embed_dim,temp_window,padding='same')

        # dropout
        self.dropout = nn.Dropout(dropout)
        # Pooling layer
        self.pool = nn.MaxPool1d(3,3)
        # Norm layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(16)
        self.bn4 = nn.BatchNorm1d(embed_dim)

    def forward(self, x):

        # TCN input video stream
        x = self.dropout(self.bn1(self.pool(F.relu(self.convL1(x))))) # b,c,L
        x = self.dropout(self.bn2(self.pool(F.relu(self.convL2(x))))) 
        x = self.dropout(self.bn3(self.pool(F.relu(self.convL3(x))))) 
        x = self.dropout(self.bn4(self.pool(F.relu(self.convL4(x))))) 
        f = torch.permute(x,(0,2,1)) # b,L,c

        return f