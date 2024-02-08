import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim

import matplotlib.pyplot as plt


torch.manual_seed(42)

### DATASETS ###

file = "Datasets/BRCA/proteinOutFileCat_nn.csv"
#file = input("Enter Dataset: ")


#file = "Datasets/BRCA/rnaSeqOutFileCat_nn.csv"
#file = "Datasets/STAD/rnaSeqOutFileCat_nn.csv"

#file = "Datasets/PAAD/rnaSeqOutFileCat_nn.csv"
#file = "Datasets/LUAD/rnaSeqOutFileCat_nn.csv"


pd_celldata = pd.read_csv(file, sep=',', header=0, index_col=0)
pd_celldata = pd_celldata.rename({'Cancer Type': 'cancer_type'}, axis=1)

features = list(pd_celldata)[:-1]

X = pd_celldata.drop('cancer_type', axis=1)
y = pd_celldata['cancer_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,
    stratify=y, random_state=0
)


def get_feature_rankings(X, y):
    rf = RandomForestClassifier()
    rf.fit(X, y)
    # Sort indices by feature importance in descending order
    sorted_indices = np.argsort(rf.feature_importances_)[::-1]
    # Initialize rankings such that the most important feature gets rank 1
    rankings = np.empty_like(sorted_indices)
    rankings[sorted_indices] = np.arange(1, len(rf.feature_importances_) + 1)
    return rankings, rf.feature_importances_

rankings, imp = get_feature_rankings(X, y)
print(rankings)


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


def get_loader(data):
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader
        
def gene_loader(X, y):
    dataset = MyDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader   


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


class CombinedModel(nn.Module):
    def __init__(self, ae, mlp):
        super(CombinedModel, self).__init__()
        self.ae = ae
        self.mlp = mlp
    
    def forward(self, x):
        recon_batch, latent_representation = self.ae(x)
        mlp_output = self.mlp(latent_representation)
        return recon_batch, mlp_output
    


def custom_weighted_mse_loss(output, target, feature_weights):
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(output, target)
    weights = torch.tensor(feature_weights, device=output.device, dtype=torch.float32)
    weighted_loss = loss * weights
    return weighted_loss.mean()

def initialize_feature_weights(num_features):
    # Initialize weights with an importance concept in mind, 1 + (1/rank)
    initial_ranks = np.arange(1, num_features + 1)
    feature_weights = 1 + (1 / initial_ranks)  # This reflects initial importance
    return feature_weights

def update_feature_weights(feature_weights, rankings, num_features):
    # Recalculate weights based on new rankings, maintaining alignment with original features
    updated_weights = np.ones(num_features)  # Start with a baseline importance
    for i, rank in enumerate(rankings):
        updated_weights[i] = 1 + (1 / rank)  # Recompute weight based on rank
    return updated_weights




def perform_sensitivity_analysis(model, data_loader, rankings, top_features_threshold=500):
    model.eval()
    num_features = len(rankings)
    feature_importances = np.zeros(num_features)
    original_loss = compute_loss(model, data_loader)

    for feature_index in range(num_features):
        rank = rankings[feature_index]
        # Check if the feature's rank is within the top features threshold
        if rank <= top_features_threshold:
            perturbed_loss = compute_loss_with_perturbation(model, data_loader, feature_index)
            improvement = original_loss - perturbed_loss
            feature_importances[feature_index] = max(improvement, 0)  # Only consider improvements, ignore deterioration

    # Generate new rankings based on updated feature importances
    updated_rankings = np.argsort(-feature_importances) + 1  # +1 to adjust rankings to be 1-based
    return updated_rankings

def compute_loss(model, data_loader):
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output, _ = model(data)
            loss = mse_loss(output, data)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def compute_loss_with_perturbation(model, data_loader, feature_index, epsilon=1e-4):
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in data_loader:
            original_data = data.clone()
            perturbed_data = original_data.to(device)
            perturbed_data[:, feature_index] += epsilon
            perturbed_data = perturbed_data.to(device)
            output, _ = model(perturbed_data)
            loss = mse_loss(output, perturbed_data)
            total_loss += loss.item()
    return total_loss / len(data_loader)


batch_size = 32
latent_dim = 64
input_dim = len(features)

output_dim = 5
batch_size = 757
learning_rate = 0.001


# Initialize AE and MLP models with increased complexity and dropout
ae = AE(input_dim, latent_dim)
mlp = MLP(latent_dim, output_dim)

# Create the combined model
model = CombinedModel(ae, mlp)

#optimizer = optim.AdamW(model.parameters(), lr=0.001)
# Initialize the learning rate scheduler
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

mlp_criterion = nn.CrossEntropyLoss()
ae_criterion = nn.MSELoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data for PyTorch
train_data = MyDataset(X_train, y_train)
train_loader = get_loader(train_data)

test_data = MyDataset(X_test, y_test)
test_loader = get_loader(test_data)


num_epochs = 15


def train_model(model, train_loader, test_loader, rankings, feature_weights, top_features_threshold=500, sensitivity_analysis_interval=1):
    plt_dict = {}
    lines = []   
    avgs = []
    x_plot = []
    y_plot = []
    num_features = len(rankings)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Start training VAE...")
    #model.train()
    for epoch in range(num_epochs):  # Assuming 10 epochs; adjust as needed
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, mlp_output = model(data)
            
            #ae_loss = ae_criterion(output, data)
            ae_loss = custom_weighted_mse_loss(output, data, feature_weights)  # Adjusted autoencoder loss
            #print("AE Loss: ", ae_loss)
            mlp_loss = mlp_criterion(mlp_output, target)
            #ae_loss = ae_criterion(output, data)
            #print("MLP Loss: ", mlp_loss)
    
            # Total loss
            loss = ae_loss + mlp_loss
            
        
            # Add any additional loss for the MLP output if required
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], AE Loss: {ae_loss.item():.4f}, MLP Loss: {mlp_loss.item():.4f}')
  
        if epoch % sensitivity_analysis_interval == 0:
            validation_loader = test_loader
            print(f"Performing sensitivity analysis at epoch {epoch}")
            new_rankings = perform_sensitivity_analysis(model, validation_loader, rankings, top_features_threshold)
            print(new_rankings)
            print(f"Updating feature weights at epoch {epoch}")
            feature_weights = update_feature_weights(feature_weights, new_rankings, num_features)
            print(feature_weights)
            rankings = new_rankings
            
    
        correct = 0
        total = 0

        with torch.no_grad():
            for X_test, y_test in test_loader:  
                X_test = X_test.view(X_test.size(0), -1)
                _, mlp_output = model(X_test)
                _, predicted = torch.max(mlp_output, 1)
                total += y_test.size(0)
                correct += (predicted == y_test).sum().item()



        accuracy = correct / total
        x_plot.append(epoch+1)
        y_plot.append(accuracy)

        # Evaluate accuracy on validation/test dataset
        print(f'Epoch [{epoch}/{num_epochs}], Combined Model Accuracy on Validation/Test dataset: {100 * accuracy:.2f}%')
    

    return x_plot, y_plot, rankings
        
        
        
# Initialize feature weights based on initial rankings
feature_weights = initialize_feature_weights(len(features))
x_plot, y_plot, final_rankings = train_model(model, train_loader, test_loader, rankings, feature_weights)
plt.plot(x_plot,y_plot, label='BRCA Protein Data')
plt.title('BRCA Protein Data')
plt.savefig('./plots/BRCA_prot.png')


# Use final rankings to identify the top N most important features
top_n = 500  # Number of top features you're interested in

ordered_genes = []
idxs = np.argsort(final_rankings)
for i in idxs[1:top_n]:
    ordered_genes.append(features[i])

with open('BRCA_top500_prot.txt', 'w') as f:
    for line in ordered_genes:
        f.write(f"{line}\n")









