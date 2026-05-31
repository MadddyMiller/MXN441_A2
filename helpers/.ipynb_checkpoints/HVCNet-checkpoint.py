import torch
import torch.nn as nn
import torch.nn.functional as F

# SK Unit 

class SKUnit(nn.Module):
    """
    Selective Kernel (SK) Unit.

    Applies multi-scale convolution (kernel sizes: 3 and 5) and learns channel-wise attention to adaptively fuse them.

    Args:
        D (int): Number of input channels.
        r (int): Reduction ratio for bottleneck.

    Input:
        x (Tensor): Shape (B, D, H, W)

    Output:
        V (Tensor): Shape (B, D, H, W)
    """
    
    def __init__(self, D=32, r=4):

        super().__init__()
        
        # Store hyperparameters
        self.D = D
        self.r = r

        # --- Split: two convolution branches ---
        # Branch 1: 3*3
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=D, out_channels=D, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: 5*5
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=D, out_channels=D, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(D),
            nn.ReLU(inplace=True)
        )

        # --- Fuse: channel reduction  ---
        d = D // r # reduced channel dimension
        self.fc1 = nn.Linear(in_features=D, out_features=d, bias=False) 
        self.bn = nn.BatchNorm1d(d)            
        self.relu = nn.ReLU(inplace=True)   

        # --- Select: compute attention weights ---
        self.fc2 = nn.Linear(in_features=d, out_features=2*D, bias=False) 

    def forward(self, x):
        B, D, H, W = x.shape  

        # --- Split ---
        U1 = self.conv3(x)          # (B, D, H, W)
        U2 = self.conv5(x)          # (B, D, H, W)

        # --- Fuse ---
        # combine branches
        U = U1 + U2 
        
        # global average pooling
        s = U.mean(dim=(2, 3))      # (B, D)
        
        # channel reduction
        z = self.fc1(s)             # (B, d)
        z = self.bn(z)
        z = self.relu(z)

        # --- Select ---
        # compute attention
        attn = self.fc2(z)              # (B, 2D)
        attn = attn.view(B, 2, D)       # (B, 2, D)
        attn = F.softmax(attn, dim=1)   # across branches

        # extract branch weights
        a = attn[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B, D, 1, 1)
        b = attn[:, 1].unsqueeze(-1).unsqueeze(-1)

        # weighted aggregation
        V = a * U1 + b * U2   # (B, D, H, W)
        
        return V
    

# MIT Module 

class MIT(nn.Module):
    """
    Multi-scale Image-to-Tokens (MIT) module.

    Converts an image into a sequence of tokens using CNN feature extraction + patch embedding.

    Args:
        d (int): Token embedding dimension.
        D (int): CNN feature channels.
        P (int): Patch size.

    Input:
        x (Tensor): Shape (B, 3, H, W)

    Output:
        x (Tensor): Shape (B, N+1, d)
    """
    
    def __init__(self, d=384, D=32, P=4):

        super().__init__()

        # Store hyperparameters
        self.P = P 
        self.d = d  
        self.D = D 

        # --- Initial convolution + downsampling ---
        self.conv = nn.Conv2d(in_channels=3, out_channels=D, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(D)                                                    
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Selective Kernel ---
        self.sk = SKUnit(D, r=4)

        # --- Patch extraction ---
        self.unfold = nn.Unfold(kernel_size=P, stride=P) # extract non-overlapping P*P patches
        
        # --- Patch projection ---
        self.proj = nn.Linear(in_features=P*P*D, out_features=d) # map each patch to token embedding
    
        # add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))

    def forward(self, x):
        B = x.shape[0] # extract batch size

        # --- Initial convolution + downsampling ---
        x = self.conv(x)        # (B, D, H/2, W/2)
        x = self.bn(x)
        x = self.pool(x)        # (B, D, H/4, W/4)

        # --- SK unit ---
        x = self.sk(x)

        # --- Patch extraction ---
        x = self.unfold(x)      # (B, D*P*P, N)
        x = x.transpose(1,2)    # (B, N, D*P*P)

        # --- Patch projection ---
        x = self.proj(x)        # (B, N, d)

        # create class token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
        # prepend CLS token to patch tokens
        x = torch.cat([cls, x], dim=1)          #  (B, N+1, d)

        return x 
    


# Mixconv Module 

class MixConv(nn.Module):
    """
    Mixed Depthwise Convolution (MixConv).

    Splits input channels into equal groups, applies depthwise convolution
    with a distinct kernel size to each group, and then concatenates the results.

    Args:
        in_channels (int): Number of input channels (C).
        kernel_sizes (list[int]): Kernel sizes per group (e.g. [3, 5, 7]).

    Input:
        x (Tensor): Shape (B, C, H, W)

    Output:
        Y (Tensor): shape (B, C, H, W)
    """

    def __init__(self, in_channels=512, kernel_sizes=(3, 5, 7)):
        super().__init__()

        # Store hyperparameters
        self.in_channels = in_channels
        self.kernel_sizes = list(kernel_sizes)
        num_groups = len(self.kernel_sizes)

        # --- Channel-wise partitioning ---
        # Split channels evenly across groups
        splits = [in_channels // num_groups] * num_groups
        self.splits = splits

        # --- Multi-scale depth-wise convolution ---
        self.convs = nn.ModuleList()
        for c, k in zip(self.splits, self.kernel_sizes):
            self.convs.append(
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=k,
                    padding= (k-1) // 2,   
                    groups=c, # depth-wise convolution
                    bias=True
                )
            )

    def forward(self, x):

        # --- Partition input into groups ---
        x_groups = torch.split(x, self.splits, dim=1)

        # --- Apply multi-scale depth-wise convolution ---
        # (different kernel size for each group)
        y_groups = []
        for conv, xi in zip(self.convs, x_groups):
            y_groups.append(conv(xi))

        # --- Concatenate group outputs ---
        Y = torch.cat(y_groups, dim=1)

        return Y
    

# MCF Module 

class MCF(nn.Module):
    def __init__(self, dim, expansion=4):
        super().__init__()
        dim2 = dim * expansion
        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)  # added
        
        self.conv1 = nn.Conv2d(dim, dim2, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(dim2)  # added
        
        self.mixconv = MixConv(dim2)
        self.norm3 = nn.BatchNorm2d(dim2)  # added
        
        self.conv2 = nn.Conv2d(dim2, dim, kernel_size=1)
        
        self.act = nn.GELU()
        
    def forward(self, x):
        B, N, C = x.shape
        
        cls_token = x[:, :1]
        x = x[:, 1:]
        
        num_patches = x.shape[1]
        H = W = int(num_patches ** 0.5)
        
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        x = self.dwconv(x)
        x = self.norm1(x)  # added
        x = self.act(x)
        
        x = self.conv1(x)
        x = self.norm2(x)  # added
        x = self.act(x)
        
        x = self.mixconv(x)
        x = self.norm3(x)  # added
        x = self.act(x)
        
        x = self.conv2(x)
        
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([cls_token, x], dim=1)
        return x
    

# MSA Module 
class MSA(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(
            dim,
            heads ,
            batch_first=True
        )

    def forward(self, x):
        out, attn = self.attn(
            x, x, x,
            need_weights=True,
            average_attn_weights=False  
        )
        

        return out, attn


# Encoder Layer 
class EncoderLayer(nn.Module):
    """
    Encoder layer, comprised of a Multihead Self-Attention Module and a Mixed Convolutional Feed-forward module. 

    Args:
        

    Input:
        

    Output:
        
    """
    def __init__(self, dim, heads = 8):
        super().__init__()
            
        self.norm1 = nn.LayerNorm(dim)
        self.msa = MSA(dim, heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mcf = MCF(dim)

    def forward(self, x):
        # Run MSA Layer 
        attn_out, attn_map = self.msa(self.norm1(x))

        # *** CHECK
        #print(attn_map.shape)
        
        x = x + attn_out

        # Run MCF Layer 
        x = x + self.mcf(self.norm2(x))
        return x, attn_map
        
    
class FinalEncoderLayer(nn.Module):

    def __init__(self, dim, heads):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.msa = MSA(dim, heads)

    def forward(self, x):

        attn_out, attn_map = self.msa(self.norm(x))

        x = x + attn_out

        return x, attn_map
    

# MFS 
class MFS(nn.Module):
    def __init__(self, M):
        super().__init__()
        self.M = M

    def forward(self, attn, tokens):
        """
        Multi-layer Feature Selection Module.
        
        Selects top M discriminative tokens based on Hadamard product 
        of CLS attention across all heads.
        
        Args:
            attn: (B, num_heads, N+1, N+1) - multi-head attention weights from ONE layer
            tokens: (B, N+1, dim) - token features (CLS + patches)
        
        Returns:
            chosen: (B, M, dim) - selected patch tokens
        """
        B, num_heads, N_plus_1, _ = attn.shape
        N = N_plus_1 - 1  # Number of patch tokens
        
        # --- Equation 5: Extract CLS attention from each head ---
        # For each head i, get ail = [ai0, ai1, ..., aiN]
        # This is the CLS token's (index 0) attention to all tokens
        cls_attn_per_head = attn[:, :, 0, :]  # (B, num_heads, N+1)
        
        # Remove CLS-to-CLS attention (we only want CLS-to-patch)
        cls_attn_per_head = cls_attn_per_head[:, :, 1:]  # (B, num_heads, N)
        
        # --- Equation 4: Hadamard product across K heads ---
        # Al = a0l ⊙ a1l ⊙ ... ⊙ a(K-1)l
        cls_attn_aggregated = cls_attn_per_head[:, 0, :]  # Start with head 0: (B, N)
        
        for h in range(1, num_heads):
            cls_attn_aggregated = cls_attn_aggregated * cls_attn_per_head[:, h, :]  # Element-wise multiply
        
        # --- Normalization (for numerical stability) ---
        # Not explicitly in paper, but necessary to prevent vanishing
        cls_attn_aggregated = cls_attn_aggregated / (cls_attn_aggregated.sum(dim=-1, keepdim=True) + 1e-8)
        
        # --- Equation 6: Select top M tokens ---
        # "sort A0l and pick M tokens with the highest values"
        topk_indices = torch.topk(cls_attn_aggregated, self.M, dim=-1).indices  # (B, M)
        
        # --- Gather selected tokens ---
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(1)  # (B, 1)
        patch_tokens = tokens[:, 1:, :]  # Exclude CLS token: (B, N, dim)
        chosen = patch_tokens[batch_idx, topk_indices]  # (B, M, dim)
        
        return chosen
    

# HVC Net 
class HVC(nn.Module):
    def __init__(
        self,  
        dim = 384, 
        depth=10,
        heads=8,
        num_classes= 200,
        M=10
    ):
        super().__init__()
        # MIT Module 
        self.mit = MIT(d=dim)

        # Encoder Block for intermediate layers
        self.feature_layers = nn.ModuleList([
            EncoderLayer(dim, heads)
            for _ in range(depth - 1)
        ])

        # Final encoder layer
        self.final_layer = FinalEncoderLayer(dim, heads)
        
        self.mfs = MFS(M)

        self.head = nn.Linear(dim, num_classes)
        
        # ✅ ADD THIS LINE: Initialize weights
        self._init_weights()
    
    # ✅ ADD THIS ENTIRE METHOD:
    def _init_weights(self):
        """
        Initialize model weights using best practices for transformers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization for linear layers
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, nn.Conv2d):
                # Kaiming initialization for conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.BatchNorm1d)):
                # Batch/Layer norm: weight=1, bias=0
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
        
        # Special initialization for classification head
        # (smaller std for final layer helps stability)
        nn.init.trunc_normal_(self.head.weight, std=0.01)
        nn.init.constant_(self.head.bias, 0)
        
        # Initialize CLS token in MIT module
        nn.init.trunc_normal_(self.mit.cls_token, std=0.02)

    def forward(self, x):
        ### Run MIT to generate patches 
        x = self.mit(x)

        # Running Encoder layers
        selected_tokens = []

        for layer in self.feature_layers:
            x, attn = layer(x)
            selected = self.mfs(attn, x)
            selected_tokens.append(selected)

        cls = x[:, :1]

        fused = torch.cat([cls] + selected_tokens, dim=1)
        # outputting Results 
        fused, _ = self.final_layer(fused)
        out = self.head(fused[:, 0])

        return out