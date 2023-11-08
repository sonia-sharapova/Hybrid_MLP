import math
import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(42)


####### DATASET #######

pd_celldata = pd.read_csv("proteinOutFileCat_nn.csv", sep=',', header=0)
#pd_celldata = pd.read_csv("rnaSeqOutFileCat_nn.csv", sep=',', header=0)
pd_celldata = pd_celldata.rename({'Cancer Type': 'cancer_type'}, axis=1)

pd_celldata_t = pd_celldata.drop(columns="Composite.Element.REF")
#pd_celldata_t = pd_celldata.drop(columns="Hugo_Symbol")

celldata_t = pd_celldata.to_numpy()
features = list(pd_celldata)

numlist=[]
labels = []
for ind in range(celldata_t.shape[0]):
    x = celldata_t[ind,:]
    numlist.append(x)
    labels.append(x[-1])

numlist = np.array(numlist)



##### Train/ Test Split ####
from sklearn.model_selection import train_test_split

X, y = pd_celldata_t.drop(labels='cancer_type', axis=1),pd_celldata_t['cancer_type']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    stratify=y, random_state=0
)



####### Data Normalization #######

X_train_norm1 = X_train.div(X_train.sum(axis=1), axis=0)
X_train_norm2 = X_train.div(X_train.sum(axis=0), axis=1)
X_train_norm2=X_train_norm2.fillna(0)

sc.settings.verbosity = 3  # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor="white")



####### to get adata #######

X_train_a = np.array(X_train_norm1)
adata = ad.AnnData(X_train_a)
#adata.uns["name"] = "rna_seq"
adata.uns["name"] = "prot"

y_train_str = []
for i in y_train:
    y_train_str.append(str(i)) # convert to strings so that they can be recognized by scanpy
adata.obs['true_labels'] = y_train_str



######### Load Data and Preprocessing #######

class MyDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        X = self.X_train.iloc[idx].values  # Extract values from X_train DataFrame row
        y = self.y_train.iloc[idx]  # Extract label from y_train DataFrame
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long)  # Convert to PyTorch tensors

def preprocess(adata):
    sc.pp.filter_genes(adata, min_cells=1)
    sc.pp.normalize_total(adata)
    sc.pp.neighbors(adata, use_rep='X')
    return adata



######### Load Models #########


    

# Define an Autoencoder (AE) model 
class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4*1024),
            nn.ReLU(),
            nn.Linear(4*1024, 4*512),
            nn.ReLU(),
            nn.Linear(4*512, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 4*512),
            nn.ReLU(),
            nn.Linear(4*512, 4*1024),
            nn.ReLU(),
            nn.Linear(4*1024, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        latent_representation = self.encoder(x)
        recon_batch = self.decoder(latent_representation)
        return recon_batch, latent_representation


# Define a Multi-Layer Perceptron (MLP) model 
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer
        self.fc3 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc3(x)
        return x
    

# Combine AE and MLP models
class CombinedModel(nn.Module):
    def __init__(self, ae, mlp):
        super(CombinedModel, self).__init__()
        self.ae = ae
        self.mlp = mlp
    
    def forward(self, x):
        recon_batch, latent_representation = self.ae(x)
        mlp_output = self.mlp(latent_representation)
        return recon_batch, mlp_output



######### Hyperparameters #########
latent_dim = 16
input_dim = len(adata.var)
output_dim = 5
batch_size = 757
learning_rate = 0.0001

num_epochs = 20

# Initialize AE and MLP models with increased complexity and dropout
ae = AE(input_dim, latent_dim)
mlp = MLP(latent_dim, output_dim)

# Create the combined model
combined_model = CombinedModel(ae, mlp)

autoencoder_criterion = nn.MSELoss()
mlp_criterion = nn.CrossEntropyLoss()

optimizer_combined = optim.AdamW(combined_model.parameters(), lr=0.001)

# Initialize the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer_combined, step_size=5, gamma=0.5)



######### DATA LOADERS #########

X_df_train = pd.DataFrame(X_train)
y_df_train = pd.Series(y_train)

dataset_train = MyDataset(X_df_train, y_df_train)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

X_df_test = pd.DataFrame(X_test)
y_df_test = pd.Series(y_test)

dataset_test = MyDataset(X_df_test, y_df_test)
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)



######### TAINING #########

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        X_train = X_train.view(X_train.size(0), -1)

        optimizer_combined.zero_grad()
        recon_batch, mlp_output_train = combined_model(X_train)

        # Calculate AE loss (reconstruction loss)
        #ae_loss = custom_loss(recon_batch, data)
        ae_loss = nn.MSELoss()(recon_batch, X_train)
        
        # Calculate MLP loss 
        mlp_loss = nn.CrossEntropyLoss()(mlp_output_train, y_train)
      
        # Total loss
        total_loss = 10*ae_loss + mlp_loss

        optimizer_combined.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer_combined.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], AE Loss: {ae_loss.item():.4f}, MLP Loss: {mlp_loss.item():.4f}')

    # Step the learning rate scheduler
    #scheduler.step()

    # Evaluate accuracy on validation/test dataset
    combined_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_test, y_test in test_loader:  
            X_test = X_test.view(X_test.size(0), -1)
            _, mlp_output = combined_model(X_test)
            _, predicted = torch.max(mlp_output, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()

    accuracy = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Combined Model Accuracy on Validation/Test dataset: {100 * accuracy:.2f}%')
