HOME_PATH = "." # path where you cloned musicfm

import matplotlib.pyplot as plt
import os
import numpy as np
import json
from tqdm import tqdm
import sys
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.preprocessing import  LabelEncoder

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, top_k_accuracy_score
sys.path.append(HOME_PATH)
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\\..*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*TorchCodec.*|.*TorchAudio.*|.*streaming_media_decoder.*")
from musicfm.model.musicfm_25hz import MusicFM25Hz
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 200

def load_audio(path, normalize=True):
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\\..*")
            return torchaudio.load(path, normalize=normalize)
    except Exception:
        return torchaudio.load(path, normalize=normalize)

def load_dataset(json_path, base_dir="./"):
    with open(json_path, "r") as f:
        files = json.load(f)
    
    X, y = [], []
    for file in tqdm(files):
        label = file.split("/")[2]  # e.g. "./train_val/aerosmith/..."
        filename = file.split('/')[-1]
        title, ext = os.path.splitext(filename)
        tqdm.write(f"{label} / {title}")
        X.append(os.path.join(base_dir, file))
        y.append(label)
    
    return np.array(X), np.array(y)

class MelSpecDataset(Dataset):
    def __init__(self, file_paths, labels, label_encoder):
        self.file_paths = file_paths
        self.labels = label_encoder.transform(labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Assuming mel spectrograms are saved as numpy arrays
        filename = os.path.splitext(os.path.basename(self.file_paths[idx]))[0]
        audio, sample_rate = load_audio(self.file_paths[idx], normalize=True)
        num_samples = 120 * sample_rate
        if audio.shape[1] > num_samples:
            audio = audio[:, :num_samples]
        elif audio.shape[1] < num_samples:
            pad_length = num_samples - audio.shape[1]
            padding = torch.zeros((audio.shape[0], pad_length))
            audio = torch.cat((audio, padding), dim=1)
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)
        return audio, label_tensor, filename

def pad_collate(batch):
    audios, labels,filenames = zip(*batch)

    # find max length
    max_len = max(a.shape[1] for a in audios)

    padded = []
    for a in audios:
        # ensure tensor
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        c, l = a.shape
        if l < max_len:
            pad = a.new_zeros((c, max_len - l))
            a = torch.cat([a, pad], dim=1)
        padded.append(a)

    batch_audio = torch.stack(padded, dim=0)
    # labels might already be tensors
    batch_labels = torch.stack([lbl if isinstance(lbl, torch.Tensor) else torch.tensor(lbl) for lbl in labels], dim=0)
    return batch_audio, batch_labels, filenames
class Classifier(nn.Module):
    def __init__(self, time=500, channel=1024, n_classes=20):
        super(Classifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flat = nn.Flatten()
        self.FC1 = nn.Linear(channel, 128)
        self.FC2 = nn.Linear(128, n_classes)
        self.drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.avgpool(x)
        x = x.permute(0, 2, 1)
        x = self.flat(x)
        x = self.drop(self.relu(self.FC1(x)) )
        x = self.FC2(x) 
        return x
if __name__ == "__main__":
    # Use 'spawn' start method so CUDA can be initialized safely in worker processes
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # start method was already set; ignore
        pass

    x_train, y_train = load_dataset('hw1/artist20/train.json', 'hw1/artist20/')
    x_valid, y_valid = load_dataset('hw1/artist20/val.json', 'hw1/artist20/')
    le = LabelEncoder()
    le.fit(y_train)
    train_dataset = MelSpecDataset(x_train, y_train, le)
    valid_dataset = MelSpecDataset(x_valid, y_valid, le)

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "6"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=pad_collate)

    # load MusicFM model here (only once, in the main process) to avoid
    # duplicating GPU memory across DataLoader worker processes
    musicfm = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(HOME_PATH, "musicfm", "data", "msd_stats.json"),
        model_path=os.path.join(HOME_PATH, "musicfm", "data", "pretrained_msd.pt"),
    )
    if torch.cuda.is_available():
        musicfm = musicfm.to(device)
    globals()['musicfm'] = musicfm
    model = Classifier(2000,1024,len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    acc1 = []
    acc3 = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # free cached memory at start of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for inputs, labels, filenames in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels, filenames = inputs.to(device), labels.to(device), filenames
            # Check for pre-computed embeddings
            embeddings_to_compute_indices = []
            embeddings_to_compute_inputs = []
            batch_embeddings = [None] * len(filenames)
            
            embedding_dir = './embeddings'
            os.makedirs(embedding_dir, exist_ok=True)

            for i, filename in enumerate(filenames):
                embedding_path = os.path.join(embedding_dir, f"{filename}.pt")
                if os.path.exists(embedding_path):
                    try:
                        batch_embeddings[i] = torch.load(embedding_path, map_location=device)
                    except Exception:
                        # File might be corrupted, recompute
                        embeddings_to_compute_indices.append(i)
                        embeddings_to_compute_inputs.append(inputs[i])
                else:
                    embeddings_to_compute_indices.append(i)
                    embeddings_to_compute_inputs.append(inputs[i])

            # Compute embeddings for files that didn't have them saved
            if embeddings_to_compute_inputs:
                inputs_to_process = torch.stack(embeddings_to_compute_inputs).to(device)
                with torch.no_grad():
                    computed_embs = musicfm.get_latent(inputs_to_process, layer_ix=7)

                # Save computed embeddings and fill in the batch_embeddings list
                for i, original_index in enumerate(embeddings_to_compute_indices):
                    emb_to_save = computed_embs[i].cpu()
                    save_path = os.path.join(embedding_dir, f"{filenames[original_index]}.pt")
                    torch.save(emb_to_save, save_path)
                    batch_embeddings[original_index] = computed_embs[i]

            # Stack all embeddings for the batch
            emb = torch.stack(batch_embeddings).to(device)
            
            optimizer.zero_grad()
            outputs = model(emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        model.eval()
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, filenames in valid_loader:
                inputs, labels, filenames = inputs.to(device), labels.to(device), filenames
                embeddings_to_compute_indices = []
                embeddings_to_compute_inputs = []
                batch_embeddings = [None] * len(filenames)
                
                embedding_dir = './embeddings'
                os.makedirs(embedding_dir, exist_ok=True)

                for i, filename in enumerate(filenames):
                    embedding_path = os.path.join(embedding_dir, f"{filename}.pt")
                    if os.path.exists(embedding_path):
                        try:
                            batch_embeddings[i] = torch.load(embedding_path, map_location=device)
                        except Exception:
                            # File might be corrupted, recompute
                            embeddings_to_compute_indices.append(i)
                            embeddings_to_compute_inputs.append(inputs[i])
                    else:
                        embeddings_to_compute_indices.append(i)
                        embeddings_to_compute_inputs.append(inputs[i])

                # Compute embeddings for files that didn't have them saved
                if embeddings_to_compute_inputs:
                    inputs_to_process = torch.stack(embeddings_to_compute_inputs).to(device)
                    with torch.no_grad():
                        computed_embs = musicfm.get_latent(inputs_to_process, layer_ix=7)

                    # Save computed embeddings and fill in the batch_embeddings list
                    for i, original_index in enumerate(embeddings_to_compute_indices):
                        emb_to_save = computed_embs[i].cpu()
                        save_path = os.path.join(embedding_dir, f"{filenames[original_index]}.pt")
                        torch.save(emb_to_save, save_path)
                        batch_embeddings[original_index] = computed_embs[i]

                # Stack all embeddings for the batch
                emb = torch.stack(batch_embeddings).to(device)
                outputs = model(emb)
                _, predicted = torch.max(outputs.data, 1)
                _, pred3 = torch.topk(outputs.data, 3, dim=1)
                for i, label in enumerate(labels):
                    if label in pred3[i]:
                        correct_top3 += 1
                total += labels.size(0)
                correct_top1 += (predicted == labels).sum().item()
                
        print(f'Top1 Accuracy on validation set: {100 * correct_top1 / total} %')
        acc1.append(correct_top1 / total)
        print(f'Top3 Accuracy on validation set: {100 * correct_top3 / total} %')
        acc3.append(correct_top3 / total)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    plt.figure()
    plt.plot(acc1, label='Top-1 Accuracy')
    plt.plot(acc3, label='Top-3 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig('accuracy_curve_baseline.png')
    
    all_labels = []
    all_preds = []
    model.eval()
    correct_top1=0
    correct_top3=0
    with torch.no_grad():
        for inputs, labels, filenames in tqdm(valid_loader, desc="Baseline Confusion Matrix"):
            inputs, labels, filenames = inputs.to(device), labels.to(device), filenames
            embeddings_to_compute_indices = []
            embeddings_to_compute_inputs = []
            batch_embeddings = [None] * len(filenames)
            
            embedding_dir = './embeddings'
            os.makedirs(embedding_dir, exist_ok=True)

            for i, filename in enumerate(filenames):
                embedding_path = os.path.join(embedding_dir, f"{filename}.pt")
                if os.path.exists(embedding_path):
                    try:
                        batch_embeddings[i] = torch.load(embedding_path, map_location=device)
                    except Exception:
                        # File might be corrupted, recompute
                        embeddings_to_compute_indices.append(i)
                        embeddings_to_compute_inputs.append(inputs[i])
                else:
                    embeddings_to_compute_indices.append(i)
                    embeddings_to_compute_inputs.append(inputs[i])

            # Compute embeddings for files that didn't have them saved
            if embeddings_to_compute_inputs:
                inputs_to_process = torch.stack(embeddings_to_compute_inputs).to(device)
                with torch.no_grad():
                    computed_embs = musicfm.get_latent(inputs_to_process, layer_ix=7)

                # Save computed embeddings and fill in the batch_embeddings list
                for i, original_index in enumerate(embeddings_to_compute_indices):
                    emb_to_save = computed_embs[i].cpu()
                    save_path = os.path.join(embedding_dir, f"{filenames[original_index]}.pt")
                    torch.save(emb_to_save, save_path)
                    batch_embeddings[original_index] = computed_embs[i]

            # Stack all embeddings for the batch
            emb = torch.stack(batch_embeddings).to(device)
            outputs = model(emb)
            _, predicted = torch.max(outputs.data, 1)
            _, pred3 = torch.topk(outputs.data, 3, dim=1)
            for i, label in enumerate(labels):
                if label in pred3[i]:
                    correct_top3 += 1
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_task2_baseline.png')
    
    with open('log_task2_baseline.txt', 'w') as f:
        f.write(f"  Top-1 Accuracy: {correct_top1 / total:.4f}\n")
        f.write(f"  Top-3 Accuracy: {correct_top3 / total:.4f}\n")
    