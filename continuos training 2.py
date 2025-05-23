import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import editdistance

from continuous_sign_dataset import ContinuousSignDataset, ctc_collate_fn
from encoder_with_prediction_of_a_label import PositionalEncoding, TransformerEncoderLayer

#─────────────────────────────────────────────────────────────────────────────
# 1. Continuous Sign Encoder for CTC
#─────────────────────────────────────────────────────────────────────────────
class ContinuousSignEncoder(nn.Module):
    def __init__(self, feat_dim, d_model=64, n_layers=4, nhead=4,
                 num_classes=50, dropout=0.1, max_len=1000):
        super().__init__()
        self.embedding    = nn.Linear(feat_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.layers       = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead,
                                    dim_feedforward=2*d_model,
                                    dropout=dropout)
            for _ in range(n_layers)
        ])
        self.fc_out       = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, T, feat_dim]
        x = self.embedding(x)              # [B, T, d_model]
        x = self.pos_encoding(x)           # [B, T, d_model]
        for layer in self.layers:
            x = layer(x)                   # [B, T, d_model]
        logits = self.fc_out(x)            # [B, T, num_classes]
        return logits.permute(1, 0, 2)      # [T, B, num_classes]

#─────────────────────────────────────────────────────────────────────────────
# 2. CTC Greedy Decoder for Metrics
#─────────────────────────────────────────────────────────────────────────────
def ctc_greedy_decode(log_probs: torch.Tensor, blank_idx: int, encoder) -> list:
    # log_probs: [T, C]
    preds = log_probs.argmax(dim=1).cpu().numpy().tolist()
    decoded = []
    prev = None
    for p in preds:
        if p != blank_idx and p != prev:
            decoded.append(p)
        prev = p
    return encoder.inverse_transform(decoded).tolist()

#─────────────────────────────────────────────────────────────────────────────
# 3. Training Loop with Additional Metrics
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
    # Prepare dataset
    dataset      = ContinuousSignDataset(csv_path, N=N, blank_token=blank_token)
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=ctc_collate_fn)
    val_loader   = DataLoader(dataset, batch_size=1,
                              shuffle=False, collate_fn=ctc_collate_fn)

    device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feat_dim    = next(iter(dataset.video_to_frames.values())).shape[1]
    num_classes = len(dataset.encoder.classes_)

    model     = ContinuousSignEncoder(feat_dim=feat_dim,
                                      d_model=64,
                                      n_layers=4,
                                      nhead=4,
                                      num_classes=num_classes,
                                      dropout=0.1,
                                      max_len=1000).to(device)
    ctc_loss  = nn.CTCLoss(blank=dataset.blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=scheduler_factor,
                                                     patience=scheduler_patience)

    best_loss  = float('inf')
    no_improve = 0

    for epoch in range(1, epochs+1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for inputs, targets, in_lens, tgt_lens in train_loader:
            inputs = inputs.permute(1,0,2).to(device)   # [B, T, feat]
            log_probs = model(inputs)                   # [T, B, C]
            log_probs = F.log_softmax(log_probs, dim=2)

            loss = ctc_loss(log_probs,
                            targets.to(device),
                            torch.tensor(in_lens),
                            torch.tensor(tgt_lens))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss      = 0.0
        seq_acc_sum   = 0
        token_acc_sum = 0.0
        wer_sum       = 0.0
        val_count     = 0

        with torch.no_grad():
            for inputs, targets, in_lens, tgt_lens in val_loader:
                inputs = inputs.permute(1,0,2).to(device)
                log_probs = model(inputs)
                log_probs = F.log_softmax(log_probs, dim=2)

                loss = ctc_loss(log_probs,
                                targets.to(device),
                                torch.tensor(in_lens),
                                torch.tensor(tgt_lens))
                val_loss += loss.item()

                # Decoding
                decoded = ctc_greedy_decode(log_probs.squeeze(1),
                                            dataset.blank_idx,
                                            dataset.encoder)
                # Ground truth sequence (remove blanks)
                gt_idxs   = targets.cpu().numpy().tolist()
                gt_labels = [lbl for lbl in dataset.encoder.inverse_transform(gt_idxs)
                             if lbl != blank_token]

                # Sequence accuracy
                seq_acc_sum += int(decoded == gt_labels)

                # Token accuracy
                matches = sum(p==g for p,g in zip(decoded, gt_labels))
                token_acc_sum += matches / max(len(gt_labels),1)

                # WER
                dist = editdistance.eval(decoded, gt_labels)
                wer_sum += dist / max(len(gt_labels),1)

                val_count += 1

        avg_val_loss = val_loss / val_count
        seq_acc      = seq_acc_sum / val_count
        token_acc    = token_acc_sum / val_count
        avg_wer      = wer_sum / val_count

        print(f"Epoch {epoch}/{epochs}  "
              f"Train Loss: {avg_train_loss:.4f}  "
              f"Val Loss: {avg_val_loss:.4f}  "
              f"SeqAcc: {seq_acc:.3f}  "
              f"TokenAcc: {token_acc:.3f}  "
              f"WER: {avg_wer:.3f}")

        # Scheduler & Early stopping
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_loss:
            best_loss  = avg_val_loss
            no_improve = 0
            torch.save(model.state_dict(), "best_continuous_sign_encoder_ctc.pth")
        else:
            no_improve += 1
            if no_improve >= earlystop_patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_loss:.4f}")

if __name__ == "__main__":
    csv_path = r"C:\Users\SADAT\Desktop\university\CoE_4\semester 1\Final Project\our coding files\pythonProject\video_landmarks_dataset_filled.csv"
    train_ctc(csv_path=csv_path,
              N=3,
              blank_token='<blank>',
              epochs=50,
              batch_size=4,
              lr=1e-4,
              scheduler_factor=0.5,
              scheduler_patience=2,
              earlystop_patience=5)
