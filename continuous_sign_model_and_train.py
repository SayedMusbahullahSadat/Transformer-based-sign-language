import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from continuous_sign_dataset import ContinuousSignDataset, ctc_collate_fn
from encoder_with_prediction_of_a_label import PositionalEncoding, TransformerEncoderLayer

#─────────────────────────────────────────────────────────────────────────────
# 1. Continuous Sign Encoder for CTC
#─────────────────────────────────────────────────────────────────────────────
class ContinuousSignEncoder(nn.Module):
    def __init__(self, feat_dim, d_model=64, n_layers=4, nhead=4, num_classes=50, dropout=0.1, max_len=1000):
        super().__init__()
        # Input projection
        self.embedding = nn.Linear(feat_dim, d_model)
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward=2*d_model, dropout=dropout)
            for _ in range(n_layers)
        ])
        # Output projection
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, T, feat_dim]
        x = self.embedding(x)             # [B, T, d_model]
        x = self.pos_encoding(x)          # [B, T, d_model]
        for layer in self.layers:
            x = layer(x)                  # [B, T, d_model]
        logits = self.fc_out(x)           # [B, T, num_classes]
        # CTC expects [T, B, C]
        return logits.permute(1, 0, 2)     # [T, B, num_classes]

#─────────────────────────────────────────────────────────────────────────────
# 2. Training with CTC Loss, Scheduler & Early Stopping
# CTC is a loss function and decoding approach that enables sequence-to-sequence training without needing aligned input-output pairs or an explicit decoder.
#─────────────────────────────────────────────────────────────────────────────
def train_ctc(csv_path,
              N=3,
              blank_token='<blank>',
              epochs=20,
              batch_size=4,
              lr=1e-4,
              scheduler_factor=0.5,
              scheduler_patience=2,
              earlystop_patience=5):
    # Dataset & DataLoader
    dataset = ContinuousSignDataset(csv_path, N=N, blank_token=blank_token)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn
    )

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model instantiation
    feat_dim = next(iter(dataset.video_to_frames.values())).shape[1]
    num_classes = len(dataset.encoder.classes_)
    model = ContinuousSignEncoder(
        feat_dim=feat_dim,
        d_model=64,
        n_layers=4,
        nhead=4,
        num_classes=num_classes,
        dropout=0.1,
        max_len=1000
    ).to(device)

    # Loss, optimizer, scheduler
    ctc_loss = nn.CTCLoss(blank=dataset.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience
    )

    # Early stopping setup
    best_loss = float('inf')
    no_improve = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for inputs, targets, input_lengths, target_lengths in loader:
            # inputs: [T, B, feat_dim] -> [B, T, feat_dim]
            inputs = inputs.permute(1, 0, 2).to(device)

            # Forward pass
            log_probs = model(inputs)                         # [T, B, C]
            log_probs = nn.functional.log_softmax(log_probs, dim=2)

            # Prepare targets and lengths
            targets = targets.to(device)
            input_lengths = torch.tensor(input_lengths, dtype=torch.long)
            target_lengths = torch.tensor(target_lengths, dtype=torch.long)

            # Compute CTC loss
            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Compute epoch metrics
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs}, Avg CTC Loss: {avg_loss:.4f}")

        # Step scheduler
        scheduler.step(avg_loss)

        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_continuous_sign_encoder_ctc.pth")
        else:
            no_improve += 1
            if no_improve >= earlystop_patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Avg CTC Loss: {best_loss:.4f}")
    print("Best model saved to best_continuous_sign_encoder_ctc.pth")

#─────────────────────────────────────────────────────────────────────────────
# 3. Entry Point
#─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = (
        r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\pythonProject\video_landmarks_dataset_filled.csv"
    )
    train_ctc(
        csv_path=csv_path,
        N=3,
        blank_token='<blank>',
        epochs=50,
        batch_size=4,
        lr=1e-4,
        scheduler_factor=0.5,
        scheduler_patience=2,
        earlystop_patience=5
    )
