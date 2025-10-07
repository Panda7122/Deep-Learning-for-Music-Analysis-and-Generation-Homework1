import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.preprocessing import  LabelEncoder
import torchaudio
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, top_k_accuracy_score
import glob
import gc
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\\..*")
warnings.filterwarnings("ignore", category=UserWarning, message=r".*TorchCodec.*|.*TorchAudio.*|.*streaming_media_decoder.*")
import torch.nn as nn
import torch.optim as optim
class Conv_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=1, pooling=2):
        super(Conv_2d, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, shape, stride=stride, padding=shape//2)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(pooling)
    def forward(self, x):
        out = self.mp(self.relu(self.bn(self.conv(x))))
        return out
def load_dataset(json_path, base_dir="./"):
    with open(json_path, "r") as f:
        files = json.load(f)
    
    X, y = [], []
    for file in files:
        label = file.split("/")[2]  # e.g. "./train_val/aerosmith/..."
        filename = file.split('/')[-1]
        title, ext = os.path.splitext(filename)
        # tqdm.write(f"{label} / {title}")
        X.append(os.path.join(base_dir, file))
        y.append(label)
    
    return np.array(X), np.array(y)


def load_audio(path, normalize=True):
    """Load audio while suppressing torchaudio deprecation/user warnings.

    This wraps `torchaudio.load` in a warnings.catch_warnings context so the
    deprecation messages about TorchCodec/StreamReader don't spam the output.
    If the filtered call fails for some reason, it falls back to calling
    `torchaudio.load` normally so errors are still visible.
    """
    try:
        with warnings.catch_warnings():
            # Filter warnings coming from torchaudio internals about StreamReader
            # and TorchCodec migration. Use a module regex to avoid hiding
            # unrelated warnings.
            warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio\\..*")
            return torchaudio.load(path, normalize=normalize)
    except Exception:
        return torchaudio.load(path, normalize=normalize)
def evaluate_model(clf, X_val, y_val):
    probs = clf.predict_proba(X_val)
    
    # Top-1 prediction
    top1_preds = np.argmax(probs, axis=1)
    top1_acc = accuracy_score(y_val, top1_preds)
    
    # Top-3 prediction
    top3_acc = top_k_accuracy_score(y_val, probs, k=3, labels=clf.classes_)
    
    cm = confusion_matrix(y_val, top1_preds)
    return top1_acc, top3_acc, cm

x_train, y_train = load_dataset('hw1/artist20/train.json', 'hw1/artist20/')
x_valid, y_valid = load_dataset('hw1/artist20/val.json', 'hw1/artist20/')



class AudioClassifierCNN(nn.Module):
    '''
    Choi et al. 2016
    Automatic tagging using deep convolutional neural networks.
    Fully convolutional network.
    '''
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50):
        super(AudioClassifierCNN, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # FCN
        self.layer1 = Conv_2d(1, 64, pooling=(2,2))
        self.layer2 = Conv_2d(64, 128, pooling=(2,2))
        self.layer3 = Conv_2d(128, 128, pooling=(2,2))
        self.layer4 = Conv_2d(128, 128, pooling=(2,2))
        self.layer5 = Conv_2d(128, 64, pooling=(2,2))

        # Global pooling to make the spatial dimensions fixed
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Dense
        self.dense = nn.Linear(64, n_class)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = self.spec_bn(x)

        # FCN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        # Global pool -> flatten -> dense
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.dense(x)
        # print(x)
        return x

# Create a custom Dataset
class MelSpecDataset(Dataset):
    def __init__(self, file_paths, labels, label_encoder):
        self.file_paths = file_paths
        self.labels = label_encoder.transform(labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Assuming mel spectrograms are saved as numpy arrays
        audio, sample_rate = load_audio(self.file_paths[idx], normalize=True)
        label = self.labels[idx]
        
        # Add a channel dimension and convert to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return audio, label_tensor


def pad_collate(batch):
    """Pad variable-length audio tensors in the batch along the time axis.

    Each audio tensor is expected to have shape (channels, samples). This
    collate pads all tensors in the batch to the maximum sample length and
    stacks them to shape (batch, channels, max_samples).
    """
    audios, labels = zip(*batch)

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
    return batch_audio, batch_labels

if __name__ == '__main__':
    # --- Main execution ---
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Create Datasets and DataLoaders
    train_dataset = MelSpecDataset(x_train, y_train, le)
    valid_dataset = MelSpecDataset(x_valid, y_valid, le)

    # make batch size and num_workers configurable to avoid OOM
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "6"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=pad_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=pad_collate)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifierCNN(n_class=len(le.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 200
    acc1 = []
    acc3 = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # free cached memory at start of epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation loop
        model.eval()
        correct_top1 = 0
        correct_top3 = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
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
    # Save the final model state
    torch.save(model.state_dict(), './checkpoint/task2_checkpoint.pth')
    print("Model saved to ./checkpoint/task2_checkpoint.pth")
    # Plotting the accuracy curves
    plt.figure()
    plt.plot(acc1, label='Top-1 Accuracy')
    plt.plot(acc3, label='Top-3 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.savefig('accuracy_curve.png')

    # Generate confusion matrix on the validation set
    all_labels = []
    all_preds = []
    # model.load_state_dict(torch.load('./checkpoint/task2_checkpoint.pth', map_location=device))
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(valid_loader, desc="Generating Confusion Matrix"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            _, pred3 = torch.topk(outputs.data, 3, dim=1)
            for i, label in enumerate(labels):
                if label in pred3[i]:
                    correct_top3 += 1
            total += labels.size(0)
            correct_top1 += (predicted == labels).sum().item()

    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(len(le.classes_)))
    fig, ax = plt.subplots(figsize=(15, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_task2.png')
        
    with open('log_task2.txt', 'w') as f:
        f.write(f"  Top-1 Accuracy: {correct_top1 / total:.4f}\n")
        f.write(f"  Top-3 Accuracy: {correct_top3 / total:.4f}\n")
    test_dir = './hw1/artist20/test/'
    files = sorted(glob.glob(os.path.join(test_dir, "*.mp3")))
    pred_dict = {}
    model.eval()
    for file in tqdm(files):
        audio, sample_rate = load_audio(file, normalize=True)
        
        with torch.no_grad():
            outputs = model(audio.unsqueeze(0).to(device))
        _, pred = torch.topk(outputs.data, 3, dim=1)
        pred = pred.squeeze(0)
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0]
        pred_dict[filename] = le.inverse_transform(pred.cpu().numpy()).tolist()

        with open('pred_task2.json', 'w') as f:
            json.dump(pred_dict, f, indent=4)