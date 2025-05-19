import os
import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

# Arguments definition
def parse_args():
    parser = argparse.ArgumentParser(description="Train a classification head on pre-computed PANDA embeddings.")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to the CSV file with embedding file names/IDs and labels.")
    parser.add_argument('--embeddings_dir', type=str, required=True, help="Base directory where .pt embedding files are stored.")
    parser.add_argument('--embedding_file_col', type=str, default='slide_id', help="Name of the column in CSV with embedding file names).")
    parser.add_argument('--label_col', type=str, default='isup_grade', help="Name of the column in CSV with target labels (e.g., ISUP grades).")
    parser.add_argument('--output_dir', type=str, default='./panda_embedding_finetune_output', help="Directory to save trained models and logs.")
    
    parser.add_argument('--embedding_dim', type=int, default=1024, help="Dimensionality of the pre-computed embeddings (e.g., 768 for ViT-B-16).")
    parser.add_argument('--num_classes', type=int, default=6, help="Number of target classes (e.g., 6 for ISUP grades 0-5).")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate for the optimizer (può essere più alta per un layer lineare).")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and validation.")
    parser.add_argument('--num_epochs', type=int, default=50, help="Number of epochs to train (può richiedere più epoche per un layer lineare).")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument('--device', type=str, default='cuda:0', help="Device to use ('cuda:0' or 'cpu').")
    parser.add_argument('--val_split_size', type=float, default=0.2, help="Fraction of data to use for validation.")
    parser.add_argument('--random_state', type=int, default=42, help="Random state for train/val split.")
    parser.add_argument('--early_stopping_patience', type=int, default=10, help="Number of epochs with no improvement after which training will be stopped.")

    return parser.parse_args()

# --- Dataset Class Panda Embeddings ---
class PANDAEmbeddingDataset(Dataset):
    def __init__(self, df, embeddings_dir, embedding_file_col, label_col, embedding_dim_placeholder_val):
        self.df = df
        self.embeddings_dir = Path(embeddings_dir)
        self.embedding_file_col = embedding_file_col
        self.label_col = label_col
        self.embedding_dim_placeholder_val = embedding_dim_placeholder_val


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        embedding_filename = row[self.embedding_file_col] 
        if not isinstance(embedding_filename, str):
            embedding_filename = str(embedding_filename)
        if not embedding_filename.endswith('.pt'):
            embedding_filename += '.pt' # Add extension if missing

        embedding_path_full = self.embeddings_dir / embedding_filename
        
        try:
            # Load the embedding. Assumed to be a raw PyTorch Tensor.
            embedding = torch.load(embedding_path_full, map_location='cpu') 
            
            # Ensure it's a PyTorch Tensor
            if not isinstance(embedding, torch.Tensor):
                raise TypeError(f"Loaded data from {embedding_path_full} is not a PyTorch Tensor. Found type: {type(embedding)}")
            
            if embedding.ndim == 2:
                if embedding.shape[0] == 0: 
                    print(f"Warning: Empty embedding tensor found for {embedding_filename}. Shape: {embedding.shape}")
                    return torch.zeros(self.embedding_dim_placeholder_val), torch.tensor(-1)
                embedding = torch.mean(embedding, dim=0) 
            elif embedding.ndim == 1: 
                pass 
            else: 
                 raise ValueError(f"Loaded embedding from {embedding_path_full} has unexpected ndim: {embedding.ndim} and shape: {embedding.shape}. Expecting 2D (N_patches, embedding_dim) or 1D (embedding_dim).")

        except FileNotFoundError:
            print(f"Error: Embedding file not found at {embedding_path_full}")
            return torch.randn(self.embedding_dim_placeholder_val), torch.tensor(-1) 
        except Exception as e:
            print(f"Error loading or processing embedding {embedding_path_full}: {e}")
            return torch.randn(self.embedding_dim_placeholder_val), torch.tensor(-1)

        label = int(row[self.label_col]) # Ensure labels are integers
        return embedding, torch.tensor(label, dtype=torch.long)

# --- Classification Model (head only) ---
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.hidden = nn.Linear(embedding_dim, 512)
        self.dropout = nn.Dropout(p=0.3)
        self.output = nn.Linear(512, num_classes)

    def forward(self, embeddings):
        x = self.hidden(embeddings)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.output(x)
        return logits

# --- Functions for training and validation ---
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch_num):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for i, (embeddings, labels) in enumerate(dataloader):
        embeddings, labels = embeddings.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy()) 
        all_labels.extend(labels.cpu().numpy())

        if i % 50 == 0:
            print(f"Epoch {epoch_num}, Batch {i}/{len(dataloader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    try:
        num_classes_output = logits.shape[1] # takes the number of classes from the last logits
        if num_classes_output > 2:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except Exception as e:
        print(f"Warning: Could not compute AUC in training: {e}")
        auc = float('nan')

    return avg_loss, accuracy, kappa, auc


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad(): # I gradienti non sono necessari qui
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            logits = model(embeddings)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy()) # MODIFICATO QUI (detach() non serve per via di torch.no_grad())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    try:
        num_classes_output = logits.shape[1] # Prendi il numero di classi dall'ultimo logits
        if num_classes_output > 2:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except Exception as e:
        print(f"Warning: Could not compute AUC in validation: {e}")
        auc = float('nan')

    return avg_loss, accuracy, kappa, auc


# --- Main Script ---
def main():
    args = parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_file_path = Path(args.output_dir) / "training_log_embeddings.csv"
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading data...")
    df_full = pd.read_csv(args.csv_path)

    def has_pt(row):
        fn = str(row[args.embedding_file_col])
        if not fn.endswith('.pt'):
            fn += '.pt'
        return (Path(args.embeddings_dir) / fn).exists()
    df_full = df_full[df_full.apply(has_pt, axis=1)].reset_index(drop=True)

    df_full = df_full.dropna(subset=[args.label_col])
    df_full = df_full[df_full[args.label_col].apply(lambda x: str(x).isdigit())]
    df_full[args.label_col] = df_full[args.label_col].astype(int)

    print(f"Dopo filtro: {len(df_full)} esempi validi")

    train_df, val_df = train_test_split(
        df_full, 
        test_size=args.val_split_size, 
        random_state=args.random_state, 
        stratify=df_full[args.label_col] if args.label_col in df_full else None
    )
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    print("Initializing classification head model...")
    classification_model = ClassificationHead(
        args.embedding_dim, 
        args.num_classes
    ).to(device)
    
    train_dataset = PANDAEmbeddingDataset(train_df, args.embeddings_dir, args.embedding_file_col, args.label_col, args.embedding_dim)
    val_dataset = PANDAEmbeddingDataset(val_df, args.embeddings_dir, args.embedding_file_col, args.label_col, args.embedding_dim)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.AdamW(classification_model.parameters(), lr=args.learning_rate)
    print("Optimizing the classification head parameters.")

    best_val_kappa = -1.0 
    log_data = []
    epochs_no_improve = 0 # Counter for early stopping

    # --- TensorBoard setup ---
    writer = SummaryWriter(log_dir=args.output_dir)

    print("Starting training of classification head...")
    for epoch in range(args.num_epochs):
        start_time = time.time()
        train_loss, train_acc, train_kappa, train_auc = train_one_epoch(
            classification_model, train_loader, criterion, optimizer, device, epoch+1
        )
        val_loss, val_acc, val_kappa, val_auc = validate(
            classification_model, val_loader, criterion, device
        )
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1}/{args.num_epochs} | Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}, AUC: {train_auc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}, AUC: {val_auc:.4f}")
        
        log_data.append({
            'epoch': epoch+1,
            'train_loss': train_loss, 'train_acc': train_acc, 'train_kappa': train_kappa, 'train_auc': train_auc,
            'val_loss': val_loss,   'val_acc': val_acc,   'val_kappa': val_kappa,   'val_auc': val_auc,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        pd.DataFrame(log_data).to_csv(log_file_path, index=False)

        # TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch+1)
        writer.add_scalar('Train/Accuracy', train_acc, epoch+1)
        writer.add_scalar('Train/Kappa', train_kappa, epoch+1)
        writer.add_scalar('Train/AUC', train_auc, epoch+1)        
        writer.add_scalar('Val/Loss', val_loss, epoch+1)
        writer.add_scalar('Val/Accuracy', val_acc, epoch+1)
        writer.add_scalar('Val/Kappa', val_kappa, epoch+1)
        writer.add_scalar('Val/AUC', val_auc, epoch+1)
        writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch+1)

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_model_path = Path(args.output_dir) / "best_classification_head_kappa.pth"
            torch.save(classification_model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path} (Val Kappa: {best_val_kappa:.4f})")
            epochs_no_improve = 0 # Reset the counter if there is an improvement
        else:
            epochs_no_improve += 1 # Increment the counter if there is no improvement

        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping triggered after {args.early_stopping_patience} epochs with no improvement on validation kappa.")
            break # Stop the training loop

    writer.close()
    print("Training completed.")
    if epochs_no_improve < args.early_stopping_patience:
        print(f"Completed {args.num_epochs} epochs.")
    print(f"Best validation Kappa: {best_val_kappa:.4f}")
    print(f"Training logs saved to {log_file_path}")
    print(f"Best model (classification head) saved to {Path(args.output_dir) / 'best_classification_head_kappa.pth'}")

if __name__ == '__main__':
    main()