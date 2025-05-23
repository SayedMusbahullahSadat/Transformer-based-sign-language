#%% md
# Step 1: Install/Import Dependencies
#%%
import torch
import torch.nn as nn
import math
#%% md
# 2. Load & Preprocess the CSV
#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# STEP 2.1: Load the CSV
df = pd.read_csv(r"C:\Users\SADAT\Desktop\university\CoE_4\Final Project\our coding files\PRACTICE DATA\cleaned_vedio_dataset_version2.csv")

# Inspect columns
print(df.columns)

# Example columns: 
# ['label', 'hand_1_0_x', 'hand_1_0_y', ...]
#%% md
# 2.2: Separate Labels & Features
#%%
# Drop the 'label' to get numeric columns only
feature_cols = [c for c in df.columns if c != 'label']
X = df[feature_cols].values  # shape: [num_samples, 126] if 2 hands

# Get the labels
y = df['label'].values  # shape: [num_samples]

#%% md
# 2.3: Handle Missing or Zero-Filled Hands:
#     Some rows have all zeros for the second hand. You can either:
#     Keep them as is (the model learns “one-hand” vs. “two-hands”).
#     Or set them to NaN and drop/impute them if truly missing.
# 
# For now, let’s keep them.
#%% md
# 2.4: Normalization (Scaling)
#%%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # shape remains [num_samples, 126]
#%% md
# 2.5: Encode Labels (If Needed)
#%%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_enc = le.fit_transform(y)  # e.g., unknown -> 0, hello -> 1, ...
num_classes = len(le.classes_)

#%% md
# 3. Split into Train/Validation/Test
# Train set: ~70%
# Validation set: ~15%
# Test set: ~15%
#%%
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y_enc, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
#%% md
# 4. Create a PyTorch Dataset & DataLoader
#%%
import torch
from torch.utils.data import Dataset, DataLoader

class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()    # shape: [N, 126]
        self.y = torch.from_numpy(y).long()     # shape: [N]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SignDataset(X_train, y_train)
val_dataset   = SignDataset(X_val, y_val)
test_dataset  = SignDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)
test_loader  = DataLoader(test_dataset, batch_size=32)

#%% md
# 5. Build a Model
#%% md
# Because this is static data (just one row per sample), you have two main approaches:
# 
# (Simpler) Treat the entire 126D input as one vector, feed it into a simple MLP or a small Transformer block that sees each “landmark” as a pseudo‐time step of length 42 per hand, etc.
# (More “Transformer-like”) Reshape your data as a sequence of length 42 (if each hand has 21 landmarks, each landmark = (x, y, z) => an embedding dimension of 3?), and pass it into a 1D Transformer encoder. This can capture some relationships among landmarks.
# Below is an example Transformer that treats each landmark as a “token.” We’ll do:
# 
# For 2 hands, 42 “tokens” (21 landmarks × 2).
# Each “token” is (x, y, z) → we’ll embed from dimension 3 up to, say, 32 or 64.
#%% md
# 5.1 Reshape X to [Batch, Seq_Len, Features]
# We have 126 features per sample → each hand has 21 * 3 = 63 → 2 hands = 126 total.
# 
# Seq Length = 42 (21 landmarks × 2 hands).
# Per “token” features = 3 (x, y, z).
# Thus, each row [126] can be reshaped into [42, 3]. We’ll do this inside the model.
#%% md
# 5.2 Define the Positional Encoding
#%%
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape: [1, max_len, d_model]

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

#%% md
# 5.3 Define an Encoder-Only Transformer
#%%
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, src):
        # Self-attention
        src2, _ = self.self_attn(src, src, src)  
        src = src + src2                  # residual
        src = self.norm1(src)            # layer norm

        # Feed-forward
        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + src2                  # residual
        src = self.norm2(src)
        
        return src

class SignEncoder(nn.Module):
    def __init__(self, d_model=64, n_layers=4, nhead=4, num_classes=10, dropout=0.1):
        super().__init__()

        # We'll embed the (x,y,z) => d_model
        self.input_dim = 3
        self.seq_len   = 42  # (21 landmarks * 2 hands)
        self.d_model   = d_model

        # Project from 3 -> d_model
        self.embedding = nn.Linear(self.input_dim, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.seq_len)

        # Stack multiple encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final classification
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x shape: [batch_size, 126]
        We'll reshape to [batch_size, 42, 3].
        """
        batch_size = x.size(0)

        # 1) reshape
        x = x.view(batch_size, self.seq_len, self.input_dim)  # => [B, 42, 3]

        # 2) embed each "token"
        x = self.embedding(x)  # [B, 42, d_model]

        # 3) add positional encoding
        x = self.pos_encoding(x)  # [B, 42, d_model]

        # 4) pass through encoder layers
        for layer in self.layers:
            x = layer(x)  # [B, 42, d_model]

        # 5) global average pool => [B, d_model]
        # for AdaptiveAvgPool1d, we need [B, d_model, 42]
        x = x.transpose(1, 2)  # => [B, d_model, 42]
        x = self.global_pool(x).squeeze(-1)  # => [B, d_model]

        # 6) classification
        logits = self.fc_out(x)  # => [B, num_classes]
        return logits
        
# Note: We use LayerNorm after each sub-layer (pre–post combos vary by implementation). The above has residual → norm structure, which is a common “post-norm” layout.
#%% md
# 6. Initialize & Train the Model
#%%
num_classes = num_classes  # from label encoder
model = SignEncoder(
    d_model=64, 
    n_layers=4, 
    nhead=4, 
    num_classes=num_classes, 
    dropout=0.1
)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)      # [batch_size, num_classes]
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            # Accuracy
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}], "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.4f}")

#%%
torch.save(model.state_dict(), "sign_language_encoder_model1000.pth")
print("✅ Model saved successfully!")
#%% md
# 6.1 tunnig model for better accuray 
#%%
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# Let's assume you have:
# 1) model = SignEncoder(...)
# 2) train_loader, val_loader
# 3) device, etc.
num_classes = num_classes  # from label encoder
model = SignEncoder(
    d_model=64, 
    n_layers=4, 
    nhead=4, 
    num_classes=num_classes, 
    dropout=0.1
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the previously trained weights
model.load_state_dict(torch.load("sign_language_encoder_model1000.pth"))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Use a learning rate scheduler that reduces LR when validation acc plateaus
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2, verbose= False
)

# Early stopping parameters
patience = 5
best_val_acc = 0.0
no_improve_count = 0

num_epochs = 20

best_model_wts = copy.deepcopy(model.state_dict())

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
    
    val_acc = correct / total
    avg_val_loss = val_loss / len(val_loader)
    avg_train_loss = total_loss / len(train_loader)

    # Update LR scheduler (monitor val_acc, so mode='max')
    scheduler.step(val_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val Acc: {val_acc:.4f}, "
          f"Current LR: {optimizer.param_groups[0]['lr']}")

    # Early stopping logic
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print("Early stopping triggered.")
            break

# Load the best weights and save
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "best_encoder_model1000.pth")
print(f"Best Validation Accuracy: {best_val_acc:.4f}")




