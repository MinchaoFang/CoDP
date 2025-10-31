import pandas as pd
import numpy as np
from sympy import sequence
from transformers.models.esm.modeling_esm import EsmForMaskedLM
from transformers import AutoTokenizer
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
import torch.optim as optim

import wandb
#wandb.init(settings=wandb.Settings(mode="offline"))
api_key = "YOUR_WANDB"
#
wandb.login(key=api_key)
from torch.optim.lr_scheduler import StepLR
import time
from tqdm import tqdm
import random


import os

import shutil

from model import ContactModel

from traing_utility import *



class CSVDataset_for_validation(Dataset):
    def __init__(self):
        # transform CSV data into groups based on 'source' column

        self.data = pd.read_csv('./validation_backbone/path_replace.csv')
        
        # extract source from file_path

        self.data['source'] = self.data['file_path'].apply(lambda x: '_'.join(x.split('/')[-1].split('_')[:-1]))
        
        # reinforce protein_length column

        #self.data = self.data.sort_values(by='protein_length').reset_index(drop=True)
        
        # group by 'source'
        self.grouped_data = self.data.groupby('source')

        # get list of groups

        self.groups = list(self.grouped_data.groups.keys())

    def __len__(self):
        # return number of unique sources
        return len(self.groups)
    
    def __getitem__(self, idx):
        # access data for a specific source

        source = self.groups[idx]
        group = self.grouped_data.get_group(source)  # DataFrame for this source

        CBs = []
        seqs = []
        cb_masks = []
        backbones = []
        for _, row in group.iterrows():
            file_path = row["file_path"]
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            cb_mask = torch.ones(data["CB"].shape[0], dtype=torch.float32)
            CBs.append(torch.tensor(data["CB"], dtype=torch.float32))
            seqs.append(data["sequence"])
            backbones.append(torch.tensor(clean_backbone_coords(data["backbone_coords"]), dtype=torch.float32))
            cb_masks.append(cb_mask)
        CBs = torch.stack(CBs)
        cb_masks = torch.stack(cb_masks)
        backbones = torch.stack(backbones)
        backbones = compute_rbf(backbones)
        return CBs, seqs, cb_masks, backbones
    
class CSVDataset(Dataset):
    def __init__(self, min_length=32,crop_size=256):
        # read CSV file
 
        df= pd.read_csv("./preprocessed_all/processed_file_paths.csv")
        
        self.data = df[df["protein_length"] > min_length].dropna()
        df= pd.read_csv('./contrastive_dataset_backbone/path_replace.csv')
        #df = df[(df["pos_plddt"] >= 85) & (df["pos_rmsd"] <= 1.5)& (df["plddt_plus"] >= 1.5)
        df['protein_length'] = df["pos"].apply(len)
        self.contract_data =df[(df["protein_length"] >= min_length) & (df["protein_length"] <= crop_size)]
        
        self.data = self.data.sort_values(by='protein_length').reset_index(drop=True)
        self.contract_data = self.contract_data.drop_duplicates(subset=['pos', 'neg'], keep='first')
        self.contract_data = self.contract_data.dropna()
        print("contact data",len(self.data))
        print("contrast data",len(self.contract_data))

        self.contract_data = self.contract_data.sort_values(by='protein_length').reset_index(drop=True)
        self.data = self.data.sample(n=len(self.contract_data), random_state=42)
        self.len1 = len(self.data)
        self.len2 = len(self.contract_data)
        self.max_len = min(self.len1, self.len2)
        print(self.max_len)

    def __len__(self):
        return self.max_len
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx % self.len1]
        row_contract = self.contract_data.iloc[idx % self.len2]
        file_path = row["file_path"]
        true_contact_file = row_contract["file_path"]
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        with open(true_contact_file, 'rb') as f:
            true_contact_file_data = pickle.load(f)
        return data["CB"], data["sequence"], row_contract["pos"], row_contract["neg"], true_contact_file_data["CB"], true_contact_file_data["backbone_coords"]

class BatchSampler(Sampler):
    def __init__(self, lengths, batch_size):
        self.lengths = lengths
        self.batch_size = batch_size
        # Store original indices to ensure all indices are used
        self.indices = list(range(len(self.lengths)))
        
    def __iter__(self):
        # Sort indices by lengths
        indices = self.indices.copy()
        indices.sort(key=lambda i: self.lengths[i])
        
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if not batch:  # Skip empty batches
                continue
            
            batches.append(batch)
        
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

class CustomCollateFn:
    def __init__(self, max_length, num_bins):
        self.max_length = max_length

        self.num_bins = num_bins

        self.device = torch.device("cpu")  # Initialize on CPU, will move data to GPU later

    def __call__(self, batch):
        """
        Collates a batch of data, padding and clipping as necessary.

        Args:
            batch (List[Tuple[Tensor, str]]): A list of tuples containing CB and sequences.

        Returns:
            Tuple[Tensor, List[str], Tensor]: Padded CB, clipped sequences, and CB mask.
        """
        CBs, sequences, pos, neg, true_contact_CB, true_contact_backbone = zip(*batch)

        # Convert all CBs to torch tensors if they are numpy arrays

        CBs = [torch.tensor(cb, dtype=torch.float32) if isinstance(cb, np.ndarray) else cb for cb in CBs]

        # Find the max length in the current batch

        current_max_length = min(max(cb.shape[0] for cb in CBs), self.max_length)

        # Process CB: Pad or clip to the current_max_length

        padded_CBs = []
        cb_masks = []
        for cb in CBs:
            current_size = cb.shape[0]
            if current_size > current_max_length:
                # Clip the CB if larger than the current max length

                cb = cb[:current_max_length, :]
            elif current_size < current_max_length:
                # Pad the CB with zeros if smaller than the current max length

                padded_cb = torch.zeros(
                    current_max_length, cb.shape[1], dtype=torch.float32

                )
                padded_cb[:current_size, :] = cb

                cb = padded_cb

            padded_CBs.append(cb)

            # Create mask: 1 for valid positions, 0 for padded positions

            cb_mask = torch.zeros(current_max_length, dtype=torch.float32)
            cb_mask[:current_size] = 1.0

            cb_masks.append(cb_mask)

        # Stack the padded CBs and masks into single tensors

        padded_CBs = torch.stack(padded_CBs)
        cb_masks = torch.stack(cb_masks)

        # Clip sequences to fit the current max length

        clipped_sequences = [seq[:current_max_length] for seq in sequences]
        
        
        true_contact = [torch.tensor(cb, dtype=torch.float32) if isinstance(cb, np.ndarray) else cb for cb in true_contact_CB]

        # Find the max length in the current batch

        current_max_length = min(max(cb.shape[0] for cb in true_contact), self.max_length)

        # Process CB: Pad or clip to the current_max_length

        padded_CBs_true_contact = []
        cb_masks_true_contact = []
        for cb in true_contact:
            current_size = cb.shape[0]
            if current_size > current_max_length:
                # Clip the CB if larger than the current max length

                cb = cb[:current_max_length, :]
            elif current_size < current_max_length:
                # Pad the CB with zeros if smaller than the current max length

                padded_cb = torch.zeros(
                    current_max_length, cb.shape[1], dtype=torch.float32

                )
                padded_cb[:current_size, :] = cb

                cb = padded_cb

            padded_CBs_true_contact.append(cb)

            # Create mask: 1 for valid positions, 0 for padded positions

            cb_mask = torch.zeros(current_max_length, dtype=torch.float32)
            cb_mask[:current_size] = 1.0

            cb_masks_true_contact.append(cb_mask)

        # Stack the padded CBs and masks into single tensors

        padded_CBs_true_contact = torch.stack(padded_CBs_true_contact)
        cb_masks_true_contact = torch.stack(cb_masks_true_contact)
        
        
        #cleaned_coords = clean_backbone_coords(true_contact_backbone[0])

        #print(f"true_contact_backbone shape: {cleaned_coords.dtype}")
        #print(f"true_contact_backbone shape: {cleaned_coords.shape}")
        true_contact_backbone = [torch.tensor(clean_backbone_coords(cb)) for cb in true_contact_backbone]
        #print(f"true_contact_backbone shape: {true_contact_backbone[0].dtype}")

        #true_contact_backbone = torch.tensor(cleaned_coords, dtype=torch.float32)

        # Find the max length in the current batch
        
        current_max_length = min(max(cb.shape[0] for cb in true_contact_backbone), self.max_length)

        # Process CB: Pad or clip to the current_max_length

        padded_CBs_true_contact_backbone = []
        cb_masks_true_contact_backbone = []
        for cb in true_contact_backbone:
            current_size = cb.shape[0]
            if current_size > current_max_length:
                # Clip the CB if larger than the current max length

                cb = cb[:current_max_length, :]
            elif current_size < current_max_length:
                # Pad the CB with zeros if smaller than the current max length

                padded_cb = torch.zeros(
                    current_max_length, cb.shape[1], cb.shape[2], dtype=torch.float32

                )
                padded_cb[:current_size, :] = cb

                cb = padded_cb

            padded_CBs_true_contact_backbone.append(cb)

            # Create mask: 1 for valid positions, 0 for padded positions

            cb_mask = torch.zeros(current_max_length, dtype=torch.float32)
            cb_mask[:current_size] = 1.0

            cb_masks_true_contact_backbone.append(cb_mask)

        # Stack the padded CBs and masks into single tensors

        padded_CBs_true_contact_backbone = torch.stack(padded_CBs_true_contact_backbone)
        cb_masks_true_contact_backbone = torch.stack(cb_masks_true_contact_backbone)
        
        true_contact_backbone = compute_rbf(padded_CBs_true_contact_backbone)

        return padded_CBs, clipped_sequences, cb_masks, pos, neg, padded_CBs_true_contact, cb_masks_true_contact , true_contact_backbone


def _rbf(D):
    device = D.device
    D_min, D_max, D_count = 2.0, 22.0, 16
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, 1, 1, 1, -1])  # Adjust shape for broadcasting
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)  # Expand last dimension for RBF
    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF

def compute_rbf(backbone):
    """
    Generate B L L D tensor from backbone data.
    
    Parameters:
    backbone (torch.Tensor): A tensor of shape [B, L, 3, 3] representing the backbone coordinates.
    
    Returns:
    torch.Tensor: A tensor of shape [B, L, L, 3 * num_rbf].
    """
    # Backbone shape: [B, L, 3, 3]
    B, L, _, _ = backbone.shape
    # Step 1: Compute pairwise distances for each atom (N, CA, C)
    D_N = torch.sqrt(torch.sum((backbone[:, :, None, 0, :] - backbone[:, None, :, 0, :]) ** 2, -1) + 1e-6)
    D_CA = torch.sqrt(torch.sum((backbone[:, :, None, 1, :] - backbone[:, None, :, 1, :]) ** 2, -1) + 1e-6)
    D_C = torch.sqrt(torch.sum((backbone[:, :, None, 2, :] - backbone[:, None, :, 2, :]) ** 2, -1) + 1e-6)
    # Step 2: Stack distances along a new dimension
    D_combined = torch.stack([D_N, D_CA, D_C], dim=-1)  # Shape: [B, L, L, 3]
    # Step 3: Apply radial basis function (RBF) transformation
    RBF = _rbf(D_combined)  # Shape: [B, L, L, 3, num_rbf]
    # Step 4: Flatten the last two dimensions
    RBF = RBF.view(B, L, L, -1)  # Shape: [B, L, L, 3 * num_rbf]
    return RBF

        
def train_batch(model, batch, criterion, bins_setting, device):
    
    total_loss = 0

    batch_count = 0

    CB, sequences, CB_mask, pos, neg ,true_contact ,cb_masks_true_contact,true_contact_backbone = batch
    # Ensure sequences are List[str]
    sequences = [str(seq) for seq in sequences]
    # Move to device

    CB = CB.to(device)
    ## Clear gradients
    #
    ## Forward pass
    outputs = model(sequences, crop_size=256)
#
    ## Compute masked loss
    contact_loss= distogram_loss(outputs, CB.to(device), CB_mask.to(device), bins_setting)
    ## Backward pass and optimization
    #del outputs, CB, CB_mask
    #if len(neg) != len(neg):
    #    raise ValueError("The two lists must have the same length.")
    #if len(pos) == 0 or len(neg) == 0:
    #    raise ValueError("The two lists must not be empty.")
    #if len(pos) > 4 :
    #    indices = random.sample(range(len(neg)), 4)
    #pos = [pos[i] for i in indices]
    #neg = [neg[i] for i in indices]
    #true_contact = [true_contact[i] for i in indices]
    #cb_masks_true_contact = [cb_masks_true_contact[i] for i in indices]
        
    #true_contact,square_mask = distogram(true_contact.to(device), cb_masks_true_contact.to(device))

    combined = pos + neg
    ##print(f"Positive features shape: {positive_features}")
    true_contact_backbone = torch.cat([true_contact_backbone,true_contact_backbone], dim=0).to(device)
    contrastive_features = model(combined, crop_size=256, true_contact=true_contact_backbone )
    ##print(f"negative_features shape: {negative_features}")
    ##contrastive_loss_fn = nn.MarginRankingLoss(margin=margin).to(device)
    B =  contrastive_features.shape[0]
    target = torch.cat([torch.ones(B // 2, dtype=torch.long), torch.zeros(B // 2, dtype=torch.long)], dim=0).to(device).view(-1,1)
    ###contrastive_loss = torch.clamp(margin - (positive_features - negative_features), min=0).mean()
    contrastive_loss =  criterion(contrastive_features, target.float())
#
    loss = contact_loss + contrastive_loss
    ##  
    wandb.log({
      "contact_loss": contact_loss.item(),
        "contrastive_loss": contrastive_loss.item(),
        "item1": contrastive_features[0].item(),
        "item2": contrastive_features[B//2].item(),
        "total_loss": loss.item()
    })
    #wandb.log({
    #    "item1": contrastive_features[0].item(),
    #    "item2": contrastive_features[B//2].item(),
    #    "total_loss": loss.item()
    #})
    #wandb.log({
    #    "total_loss": loss.item()
    #    })

    return loss


import numpy as np


def validate(model, dataloader, bins_setting, device):
    """
    Validation function

    Args:
    - model: neural network model

    - dataloader: validation data loader

    - bins_setting: settings for bins in loss function

    - device: computation device (CPU/GPU)
    
    Returns:
    - Average validation loss

    - DataFrame containing sequences and their corresponding losses

    """
    model.eval()
    total_loss = 0

    batch_count = 0

    results = []  # List to store sequence and loss information

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):

            CB, sequences, CB_mask, true_contact_backbone = batch

            sequences = [str(seq) for seq in sequences]

            # Compute outputs for the entire batch
            true_contact_backbone = true_contact_backbone.to(device)
            CB = CB.to(device)
            CB_mask = CB_mask.to(device)
            outputs = model(sequences, crop_size=600, true_contact=true_contact_backbone, validation = True)

            #if outputs.shape[1] != CB.shape[1]:
            #    print(f"Dimension mismatch: outputs {outputs.shape}, CB {CB.shape}")
            #    continue

            # Compute loss for each sequence individually
            #print(f"outputs shape: {outputs.shape}")
            for i, seq in enumerate(sequences):
                output = outputs[i:i+1,:]  # Select the output for the i-th sequence
                results.append({
                    "sequence": seq,
                    "loss": output,
                })
                #print(output)

            del outputs, CB, CB_mask

    avg_loss = total_loss / max(1, batch_count)
    results_df = pd.DataFrame(results)
    
    return avg_loss, results_df

def train_model(model, train_dataloader, validation_dataloader, bins_setting, checkpoints_dir,checkpoints_to_run,epochs=50, lr=1e-4):
    """
    Complete model training pipeline

    
    Args:
    - model: neural network model

    - train_dataloader: training data loader

    - validation_dataloader: validation data loader

    - bins_setting: settings for bins in loss function

    - epochs: number of training epochs

    - lr: initial learning rate

    """
    # Initialize wandb for experiment tracking

    wandb.init(project="contact-map-prediction", config={
        "learning_rate": lr,
        "architecture": "ContactModel",
        "epochs": epochs

    })
    
    # Set device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=lr, 
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=10000, gamma=0.95)
    
    if checkpoints_to_run:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load checkpoint

        checkpoint = torch.load(checkpoints_to_run, map_location=device)


        saved_state_dict = {
            k: v for k, v in checkpoint['model_state_dict'].items()
            if k in [name for name, param in model.named_parameters() if param.requires_grad]
        }

        model.load_state_dict(saved_state_dict, strict=False)

        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    os.makedirs(checkpoints_dir, exist_ok=True)
    model.train()
    current_step = 0
    total_loss = 0
    criterion = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
    for epoch in range(epochs):
        for batch in tqdm(train_dataloader, desc="Training", leave=False):
            optimizer.zero_grad()
            loss = train_batch(model, batch, criterion, bins_setting, device)
            total_loss += loss.item()

            current_step += 1
            
            if current_step % 50000 == 0:
                print(f"Step {current_step}: Current learning rate = {optimizer.param_groups[0]['lr']}")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': average_loss,
                }, os.path.join(checkpoints_dir, f'steps_{current_step}.pth'))
            loss.backward()
            optimizer.step()
            scheduler.step()

        average_loss = total_loss / len(train_dataloader)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': average_loss,
        }, os.path.join(checkpoints_dir, f'epoch_{epoch+1}.pth'))

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {average_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        val_loss, results_df = validate(model, validation_dataloader, bins_setting, device)
        results_df_filename = os.path.join(checkpoints_dir, f'results_df_epoch_{epoch+1}.csv')
        results_df.to_csv(results_df_filename, index=False)

    
    wandb.finish()
if __name__ == "__main__":
    # Initialize the model
    bins_setting = {
        'first_break': 2.3125,
        'last_break': 21.6875,
        'num_bins': 8
    }
    crop_size = 256
    print("Model loading...")
    
    # Assuming contactModel is your defined model class
    esm2_650_path = ""
    contact_model = ContactModel(
        esm2_650_path, 
        input_channels=384, 
        n_filters=256, 
        kernel_size=3, 
        n_layers=8,
        num_bins=bins_setting['num_bins'],
        crop_size=crop_size
    )
    
    # Calculate model size statistics
    requires_grad_size = 0
    no_grad_size = 0
    for name, param in contact_model.named_parameters():
        # Calculate size in MB (assuming float32, 4 bytes per parameter)
        size_in_bytes = param.numel() * 4
        size_in_mb = size_in_bytes / (1024 ** 2)
        
        if param.requires_grad:
            requires_grad_size += size_in_mb
        else:
            no_grad_size += size_in_mb

    # print model size statistics
    print(f"Total size of weights requiring gradients: {requires_grad_size:.4f} MB")
    print(f"Total size of weights not requiring gradients: {no_grad_size:.4f} MB")
    

    train_dataset = CSVDataset(min_length=32,crop_size=crop_size)
    batch_sampler_train = CroppedLengthBucketedBatchSampler(
        lengths=train_dataset.data['protein_length'].tolist(), 
        batch_size=5,
        length_bucket_width=32,
        crop_size=crop_size
    )
    checkpoints_to_run = "" # pretrained model path
    checkpoints_dir = "" #output path
    # for backup
    copy_current_file_to_folder(checkpoints_dir)
    # set up collate_fn
    collate_fn_train = CustomCollateFn(max_length=crop_size, num_bins=bins_setting['num_bins'])

    # set up train dataloader
    train_dataloader = DataLoader(
        train_dataset, 
        num_workers=10,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn_train,
        pin_memory=True
    )

    validation_dataset = CSVDataset_for_validation()

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=1, 
        shuffle=True, 
        collate_fn=lambda x: x[0])
    # Start training
    train_model(contact_model, train_dataloader,validation_dataloader, bins_setting,checkpoints_dir,checkpoints_to_run)
