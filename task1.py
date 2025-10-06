import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, top_k_accuracy_score
import glob
import gc
def extract_logmelspec(path, mel_path, n_mels=128, sr=22050, duration=30):
    filename = path.split('/')[-1]
    title, ext = os.path.splitext(filename)
    # Load audio (truncate/pad to fixed length for consistency)
    y, _ = librosa.load(path, sr=sr, duration=duration, mono=True)
    
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel, ref=np.max)
    
    # Temporal pooling: mean & std across time
    mean = np.mean(logmel, axis=1)
    std = np.std(logmel, axis=1)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(logmel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    if not os.path.exists(f'{mel_path}'):
        os.makedirs(f'{mel_path}')
        
    plt.savefig(f'{mel_path}/{title}.png')
    plt.close() # Close the figure to free up memory
    return np.concatenate([mean, std])  # shape = 256 (128 mean + 128 std)

def load_dataset(json_path, mel_log_path, base_dir="./"):
    with open(json_path, "r") as f:
        files = json.load(f)
    
    X, y = [], []
    for file in tqdm(files):
        label = file.split("/")[2]  # e.g. "./train_val/aerosmith/..."
        filename = file.split('/')[-1]
        title, ext = os.path.splitext(filename)
        tqdm.write(f"{label} / {title}")
        feat = extract_logmelspec(os.path.join(base_dir, file), mel_log_path)
        X.append(feat)
        y.append(label)
    
    return np.array(X), np.array(y)



def evaluate_model(clf, X_val, y_val):
    probs = clf.predict_proba(X_val)
    
    # Top-1 prediction
    top1_preds = np.argmax(probs, axis=1)
    top1_acc = accuracy_score(y_val, top1_preds)
    
    # Top-3 prediction
    top3_acc = top_k_accuracy_score(y_val, probs, k=3, labels=clf.classes_)
    
    cm = confusion_matrix(y_val, top1_preds)
    return top1_acc, top3_acc, cm
x_train, y_train = load_dataset('hw1/artist20/train.json', "train_mel_log",'hw1/artist20/')
x_valid, y_valid = load_dataset('hw1/artist20/val.json', "valid_mel_log",'hw1/artist20/')
# preprocessing
encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_valid_enc = encoder.transform(y_valid)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)

models = {
    "KNN": (KNeighborsClassifier(), {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights": ["uniform", "distance"]
    }),
    "SVM": (SVC(probability=True), {
        "clf__C": [0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
        "clf__gamma": ["scale", "auto"]
    }),
    "RandomForest": (RandomForestClassifier(random_state=42), {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 10, 20]
    })
}
results = {}
for name, (model, params) in models.items():
    print(f"\nTuning {name}...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])
    grid = GridSearchCV(pipe, params, cv=3, scoring="accuracy", n_jobs=1, verbose=1)
    grid.fit(x_train, y_train_enc)

    best_model = grid.best_estimator_
    top1_acc, top3_acc, cm = evaluate_model(best_model, x_valid, y_valid_enc)

    results[name] = {
        "best_params": grid.best_params_,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "confusion_matrix": cm,
        "model": best_model
    }
    # Visualize and save the confusion matrix
    fig, ax = plt.subplots(figsize=(12, 12))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title(f'{name} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'./{name}_confusion_matrix.png')
    plt.close(fig)

    print(f"Results for {name}:")
    print(f"  Best Params: {grid.best_params_}")
    print(f"  Top-1 Accuracy: {top1_acc:.4f}")
    print(f"  Top-3 Accuracy: {top3_acc:.4f}")

# Print best model overall
best_model_name = max(results, key=lambda name: results[name]['top1_acc'])
print(f"\nBest performing model (Top-1 Acc): {best_model_name}")
print(f"  Top-1 Accuracy: {results[best_model_name]['top1_acc']:.4f}")
print(f"  Top-3 Accuracy: {results[best_model_name]['top3_acc']:.4f}")
with open('log.txt', 'w') as f:
    f.write(f"\nBest performing model (Top-1 Acc): {best_model_name}")
    f.write(f"  Top-1 Accuracy: {results[best_model_name]['top1_acc']:.4f}")
    f.write(f"  Top-3 Accuracy: {results[best_model_name]['top3_acc']:.4f}")
def load_testset(test_dir, n_mels=128, sr=22050, duration=30):
    files = sorted(glob.glob(os.path.join(test_dir, "*.mp3")))
    X, ids = [], []
    for file in tqdm(files):
        feat = extract_logmelspec(file, 'test_mel_log',n_mels=n_mels, sr=sr, duration=duration)
        X.append(feat)
        file_id = os.path.splitext(os.path.basename(file))[0]
        ids.append(file_id)
    return np.array(X), ids

X_test, test_ids = load_testset("hw1/artist20/test")
print(f"🏆 Using best model for predition: {best_model_name}")

best_model = results[best_model_name]["model"]

probs = best_model.predict_proba(X_test)
top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

pred_dict = {}
for idx, preds in zip(test_ids, top3_preds):
    labels = encoder.inverse_transform(preds)
    pred_dict[idx] = labels.tolist()
with open("pred.json", "w") as f:
    json.dump(pred_dict, f, indent=4)
