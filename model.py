import numpy as np

from transformers.models.esm.modeling_esm import EsmForMaskedLM
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class ESMEvaluator():
    def __init__(self,checkpoints_to_run,esm_name):
        bins_setting = {
        'first_break': 2.3125,
        'last_break': 21.6875,
        'num_bins': 8
        }
        crop_size = 256
        print("Model loading...")

        # Assuming contactModel is your defined model class
        self.contact_model = ContactModel(
            esm_name, 
            input_channels=384, 
            n_filters=256, 
            kernel_size=3, 
            n_layers=8,
            num_bins=bins_setting['num_bins'],
            crop_size=crop_size
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.contact_model.to(device)
        checkpoint = torch.load(checkpoints_to_run, map_location=device)
        saved_state_dict = {
            k: v for k, v in checkpoint.items()
            if k in [name for name, param in self.contact_model.named_parameters() if param.requires_grad]
        }

        self.contact_model.load_state_dict(saved_state_dict, strict=False)
    
    def predict(self,sequneces,pdb_path):
        #backbone_array = []
        #for _ in pdb_path:
        #    backbone = extract_pdb_info(_)
        #    backbone_array.append(backbone)
        #print(pdb_path)
        backbone = extract_pdb_info(pdb_path)
        #print(backbone.shape)
        backbone_with_batch = np.expand_dims(backbone, axis=0)  # shape (1, L, 3, 3)
        backbone_with_batch = np.repeat(backbone_with_batch, len(sequneces), axis=0)  # repeat batch_size to (batch_size, L, 3, 3)
        backbone_with_batch = torch.tensor(backbone_with_batch, dtype=torch.float32)
        
        backbone =  compute_rbf(backbone_with_batch)
        scores = self.contact_model(sequneces,crop_size=600, true_contact=backbone, validation = True)
        scores = scores.mean(dim=1).tolist()
        return scores


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

class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate=1, dropout_rate=0.15):
        super().__init__()
        # for padding 
        padding = (dilation_rate * (kernel_size - 1)) // 2

        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=padding, 
            dilation=dilation_rate
        )
        self.inst_norm1 = nn.InstanceNorm2d(num_features=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation_rate
        )
        self.inst_norm2 = nn.InstanceNorm2d(num_features=out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        # Add a residual connection if input and output channels differ
        self.residual_conv = None
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
            
        out = self.conv1(x)
        out = self.inst_norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.inst_norm2(out)
        out += residual
        out = F.relu(out)
        return out

class ConvPoolToFixedDim(nn.Module):
    def __init__(self, n_filters):
        super(ConvPoolToFixedDim, self).__init__()
        self.conv = nn.Conv2d(n_filters, 128, kernel_size=1)  
        self.fc = nn.Sequential(
            nn.LayerNorm(128),          # Normalization layer
            nn.Linear(128, 32),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(32, 1)          # Another fully connected layer
        )

    def forward(self, x):
        # x shape [B, L, L, 8]
        x = self.conv(x)  
        x = torch.mean(x, dim=(2, 3))    
        x = self.fc(x)
        return x

class ContactModel(nn.Module):
    def __init__(self, esm_model_name, input_channels, n_filters, kernel_size, n_layers, num_bins, crop_size):
        super().__init__()
        self.esm_model_head = EsmForMaskedLM.from_pretrained(esm_model_name)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
        
        # Freeze ESM model parameters
        for param in self.esm_model_head.parameters():
            param.requires_grad = False
        
        self.con_model = self._create_network(384, n_filters, kernel_size, n_layers)

        self.cross_projection_pair_1 =  MultiHeadCrossAttentionModule((48+num_bins),256,4)
        self.cross_projection_pair_2 =  MultiHeadCrossAttentionModule((48+num_bins),256,4)
        
        # Process through contrastive projection
        self.self_attention_projection_insert_cls = MultiHeadSelfAttention_with_cls(256,4)
        self.self_attention_projection_extract_cls = MultiHeadAttentionWithCLSToken(256,4)
        self.self_attention_projection = nn.Sequential(
            nn.LayerNorm(256),          # Normalization layer
            nn.Linear(256, 64),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(64, 1)          # Another fully connected layer
        )
        self.crop_size = crop_size
        self.num_bins = num_bins
        self.esm_mlp_z = nn.Sequential(
            nn.LayerNorm(660),          # Normalization layer
            nn.Linear(660, 128),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(128, 128)          # Another fully connected layer
        )
        self.esm_mlp_s = nn.Sequential(
            nn.LayerNorm(self.esm_model_head.config.hidden_size),          # Normalization layer
            nn.Linear(self.esm_model_head.config.hidden_size, self.esm_model_head.config.hidden_size//2),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Linear(self.esm_model_head.config.hidden_size//2, 256)          # Another fully connected layer
        )
        self.bin_projection = nn.Sequential(
            nn.InstanceNorm2d(n_filters),          # Normalization layer
            nn.Conv2d(n_filters, n_filters//4, kernel_size=5,padding=2),        # Fully connected layer
            nn.ReLU(),                   # Activation function
            nn.Conv2d(n_filters//4, num_bins, kernel_size=1)          # Another fully connected layer
        )
        
    def _create_network(self, input_channels, n_filters, kernel_size, n_layers):
        network = nn.Sequential()
        network.add_module('initial_conv', nn.Conv2d(input_channels, n_filters, kernel_size=1, padding=0))
        network.add_module('inst_norm', nn.InstanceNorm2d(n_filters))
        network.add_module('relu', nn.ReLU())

        dilation_rate = 1

        for i in range(n_layers):
            network.add_module(f'residual_block_{i}',
                               ResidualBlock2D(n_filters, n_filters, kernel_size, dilation_rate))
            dilation_rate *= 2

            if dilation_rate > 16:
                dilation_rate = 1

        return network
    
    def forward(self, sequences, crop_size=0, true_contact = None, validation = False):
        device = next(self.parameters()).device
        crop_size_current = crop_size if crop_size != 0 else self.crop_size
        #start_time = time.time()
        #print_memory_usage()
        
        # Process in chunks to reduce memory usage if needed
        with torch.no_grad():
            # Tokenize all sequences in a batch
            inputs = self.esm_tokenizer(sequences, 
                                      return_tensors='pt', 
                                      padding='longest',
                                      max_length=crop_size_current+2,  # for cls and end tokens
                                      truncation=True)
            
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # Get model outputs
            outputs = self.esm_model_head(
                input_ids=inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                output_attentions=True
            )
            del inputs
            
            # Extract hidden states (excluding special tokens)
            hidden_states = outputs.hidden_states[-1][:,1:-1,:].detach()  # Clone to make a copy
            
            # # Process attention from all layers and reshape
            all_layers_attention = torch.stack(outputs.attentions, dim=1)[:,:,:,1:-1,1:-1].detach()
            all_layers_attention = all_layers_attention.permute(0,3,4,1,2).flatten(3,4)
            del outputs
        #print(all_layers_attention.shape)
        #print_memory_usage()
        #duration = time.time() - start_time
        #print(f"ESM2 processing time: {duration:.2f}s")
        esm_z_reshape = self.esm_mlp_z(all_layers_attention)
        del all_layers_attention
        hidden_states_projection=self.esm_mlp_s(hidden_states)
        del hidden_states
        # Compute pairwise embeddings
        #print(f"Hidden states shape: {hidden_states.shape}")
        all_pair_maps = hidden_states_projection[:,:,None,:] + hidden_states_projection[:,None,:,:]
        #print(all_pair_maps.shape)
        #print_memory_usage()
        
        # Combine pair and attention maps
        combined_maps = torch.cat([all_pair_maps, esm_z_reshape], dim=-1).permute(0,3,1,2)
        
        # Free up memory for intermediate tensors
        del esm_z_reshape
        del all_pair_maps

        # Apply MLP transformation
        
        
        #print(f"Reshaped ESM features shape: {esm_z_reshape.shape}")
        #print_memory_usage()
        
        # Process through convolutional model
        conv_output = self.con_model(combined_maps)
        bin_logits = self.bin_projection(conv_output)
        #
        bin_probs = F.softmax(bin_logits, dim=1)
        bin_probs = bin_probs.permute(0, 2, 3, 1)
        del combined_maps
        del conv_output
        if true_contact is not None:
            true_contact = true_contact.to(device)
            ##print(f"Conv output shape: {all_pair_maps.shape}")
            #print(f"Bin probs shape: {bin_probs.shape}")
            
            pair_stack = torch.cat([true_contact, bin_probs], dim=3)
            #print(f"pair_stack shape: {pair_stack.shape}")
            #print(f"hidden_states_projection shape: {hidden_states_projection.shape}")
            #pair_stack = self.con_pair(pair_stack.permute(0,3,1,2))
            #print(f"pair_stack shape: {pair_stack.shape}")
            single_stack = self.cross_projection_pair_1(hidden_states_projection, pair_stack)
            single_stack = self.cross_projection_pair_2(single_stack, pair_stack.permute(0,2,1,3))
            del hidden_states_projection,pair_stack
            single_stack = self.self_attention_projection_insert_cls(single_stack)
            
            if validation:
                B = single_stack.shape[0]  # batchsize
                all_indices = torch.arange(B)
                pair_indices = torch.stack([
                    all_indices.repeat_interleave(B - 1),  # subtractor
                    torch.cat([torch.cat((all_indices[:i], all_indices[i+1:])) for i in range(B)])  
                ], dim=1)  # [C, 2]，C = B * (B - 1)

                # extract features for each pair

                features_pair_1 = single_stack[pair_indices[:, 0]]  # [C, n_filters, L, L]
                features_pair_2 = single_stack[pair_indices[:, 1]]  # [C, n_filters, L, L]

                # calculate interaction for each pair

                interaction_pair = features_pair_1 - features_pair_2 # [C, n_filters, L, L]
                # to [B, B-1]
                del features_pair_1
                del features_pair_2
                contrastive_output = self.self_attention_projection_extract_cls(interaction_pair)
                del interaction_pair
                #print(f"Contrastive output shape: {contrastive_output.shape}")
                contrastive_output = self.self_attention_projection(contrastive_output)
                contrastive_output = F.sigmoid(contrastive_output)
                contrastive_output = contrastive_output.view(B, B - 1)  # [B, B-1]
                #print(f"Contrastive output shape: {contrastive_output.shape}")
            else:
                B = single_stack.shape[0]//2
                features_pair_1 = single_stack[:B]  
                features_pair_2 = single_stack[B:]  
                interaction_pair_1 = features_pair_1 - features_pair_2 
                interaction_pair_2 = features_pair_2 - features_pair_1 
                interaction_pair = torch.cat([interaction_pair_1, interaction_pair_2], dim=0)  
                #print(f"Interaction pair shape: {interaction_pair.shape}")
                contrastive_output = self.self_attention_projection_extract_cls(interaction_pair)
                #print(f"Contrastive output shape: {contrastive_output.shape}")
                contrastive_output = self.self_attention_projection(contrastive_output)
                #print(f"Contrastive output shape: {contrastive_output.shape}")
                contrastive_output = F.sigmoid(contrastive_output)
                #print(f"Contrastive output shape: {contrastive_output.shape}")
            return contrastive_output
        #print(f"Contrastive output shape: {contrastive_output.shape}")
        else:
            return bin_probs
        
class MultiHeadCrossAttentionModule(nn.Module):
    """Enhanced multi-head cross-attention module using matrix multiplication for attention score calculation."""
    def __init__(self, contact_bins, embed_dim, num_heads):
        """
        Initialize the multi-head cross-attention module.

        :param contact_bins: int, dimensionality of contact map features
        :param embed_dim: int, embedding dimension (must be divisible by num_heads)
        :param num_heads: int, number of attention heads
        """
        super(MultiHeadCrossAttentionModule, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads  # each head dim

        # mutil-head Query, Key, Value projection

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(contact_bins, embed_dim)
        self.value_projection = nn.Linear(contact_bins, embed_dim)

        # Output projection to combine multi-head attention results.

        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # add layer normalization

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, B_L_D, B_L_L_D):
        """
        :param B_L_D: Primary input tensor with shape (B, L, D)
        :param B_L_L_D: Mapping tensor with shape (B, L, L, D)  
        :return: Output tensor with shape (B, L, D)
        """
        residual = B_L_D

        
        # projection to Q, K, V

        Q = self.query_projection(B_L_D)  # (B, L, D)
        K = self.key_projection(B_L_L_D)  # (B, L, L, D)
        V = self.value_projection(B_L_L_D)  # (B, L, L, D)
        
        # divide Q, K, V into multiple heads

        B, L, D = Q.shape

        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L, head_dim)
        K = K.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, L, L, head_dim)
        V = V.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)  # (B, num_heads, L, L, head_dim)

        # attention score calculation using matrix multiplication

        Q_expanded = Q.unsqueeze(-2)  # (B, num_heads, L, 1, head_dim)
        attention_scores = torch.matmul(Q_expanded, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, L, L, L)
        attention_scores = attention_scores.squeeze(-2)  # (B, num_heads, L, L)

        # apply softmax normalization

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, L, L)
        
        # use attention weights to weight V

        attention_weights_expanded = attention_weights.unsqueeze(-1)  # (B, num_heads, L, L, 1)
        weighted_V = attention_weights_expanded * V  # (B, num_heads, L, L, head_dim)
        context = weighted_V.sum(dim=3)  # (B, num_heads, L, head_dim)
        
        # concat heads

        context = context.transpose(1, 2).reshape(B, L, D)  # (B, L, D)
        output = self.output_projection(context)
        
        # residual connection and layer normalization

        output = self.layer_norm(output + residual)

        return output
    
class MultiHeadAttentionWithCLSToken(nn.Module):
    """Multi-head attention module that outputs only the updated CLS token."""
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadAttentionWithCLSToken, self).__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads

        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection layers

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)

        # output projection layer

        self.output_projection = nn.Linear(embed_dim, embed_dim)

        # add layer normalization and dropout

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        :param x: Input tensor with shape (B, L, D)
        :return: Updated CLS token with shape (B, D)
        """
        B, L, D = x.shape

        # Use the CLS token as Query, and all other tokens as Key and Value
        cls_token = x[:, 0:1, :]  # (B, 1, D)
        other_tokens = x[:, 1:, :]  # (B, L-1, D)

        # Project to generate Q, K, V

        Q = self.query_projection(cls_token)  # (B, 1, D)
        K = self.key_projection(other_tokens)  # (B, L-1, D)
        V = self.value_projection(other_tokens)  # (B, L-1, D)

        # split heads

        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1, head_dim)
        K = K.view(B, L - 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L-1, head_dim)
        V = V.view(B, L - 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L-1, head_dim)

        # calculate attention scores

        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, 1, L-1)

        # apply softmax normalization

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, 1, L-1)
        attention_weights = self.dropout(attention_weights)

        # use attention score to weight V

        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, 1, head_dim)

        # concat heads

        attention_output = attention_output.transpose(1, 2).contiguous().view(B, 1, D)  # (B, 1, D)

        # project output

        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)

        # residual connection and layer normalization

        cls_output = self.layer_norm(attention_output + cls_token)  # only for CLS token

        # update CLS token

        return cls_output.squeeze(1)  # (B, D)

class MultiHeadSelfAttention_with_cls(nn.Module):
    """Multi-head self-attention that incorporates a CLS token, followed by residual connection and layer normalization."""
    def __init__(self, embed_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadSelfAttention_with_cls, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # CLS token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # projection layers
        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        
        # output projection layer
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
        # add layer normalization and dropout
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        # forward feed network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        self.ffn_layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        :param x: Input tensor of shape (B, L, D)
        :return: Output tensor of shape (B, L+1, D) with CLS token included
        """
        B, L, D = x.shape
        
        # add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x_with_cls = torch.cat([cls_token, x], dim=1)  # (B, L+1, D)
        
        # self attention with residual connection
        residual = x_with_cls
        
        # generate Q, K, V
        Q = self.query_projection(x_with_cls)  # (B, L+1, D)
        K = self.key_projection(x_with_cls)    # (B, L+1, D)
        V = self.value_projection(x_with_cls)  # (B, L+1, D)
        
        # split heads
        Q = Q.view(B, L+1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L+1, head_dim)
        K = K.view(B, L+1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L+1, head_dim)
        V = V.view(B, L+1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, L+1, head_dim)
        
        # calculate attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)  # (B, num_heads, L+1, L+1)
        
        # apply softmax normalization
        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, L+1, L+1)
        attention_weights = self.dropout(attention_weights)
        
        # use attention scores to weight V
        attention_output = torch.matmul(attention_weights, V)  # (B, num_heads, L+1, head_dim)
        
        # concat heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(B, L+1, D)  # (B, L+1, D)
        
        # output projection
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)
        
        # first residual connection and layer normalization
        attention_output = self.layer_norm(attention_output + residual)
        
        # second feed forward network with residual connection
        ffn_output = self.ffn(attention_output)
        output = self.ffn_layer_norm(ffn_output + attention_output)
        
        return output

from Bio.PDB import  PDBParser
def extract_pdb_info(pdb_file):
    """
    Extract structural data from a PDB file.

    Returns:
        tuple: A tuple containing (amino_acid_sequence, cbeta_coordinates, backbone_coordinates)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    backbone_coords = []  

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != ' ':  # skip hetero residues
                    continue

                try:
                    n = residue['N'].coord
                    ca = residue['CA'].coord
                    c = residue['C'].coord
                    backbone_coords.append([n, ca, c])
                except KeyError:
                    # missing atom, skip this residue
                    continue

    return np.array(backbone_coords, dtype=np.float32)

