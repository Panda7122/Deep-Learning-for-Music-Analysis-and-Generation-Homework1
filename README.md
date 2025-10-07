# Deep-Learning-for-Music-Analysis-and-Generation-Homework1

## 41147009S 陳炫佑

## Minimum Hardware Requirements

- **CPU:** AMD Ryzen 5 7500 or equivalent and above (6 cores, 12 threads)
- **Memory:** 16GB RAM or more  
- **Storage:** At least 20GB (for datasets and models)  
- **GPU (Recommended):** NVIDIA RTX 3090 or above, at least 24GB VRAM, CUDA support  
- **Operating System:** Arch Linux x86_64(Linux 6.16.10-zen1-1-zen)
- **Python Version:** 3.12.9  

## envirement set up

1. use `git clone https://github.com/minzwon/musicfm.git` for download musicFM as baseline model.
2. download dataset to `./hw1`, the directory structure is looklike:

```text
./hw1
├── artist20
│   ├── test/
│   ├── train_val/
│   ├── train.json
│   └── val.json
├── count_score.py
├── readme.md
├── test_ans.json
└── test_pred.json
```

3. download python 3.12.9 and open a new envirment for it.
4. in envirment, use `pip3 install -r ./requirements.txt` for install needed module.

## task 1
it will run knn, random forest and svm, and use grid search cv for optimize parameter

run `python3 task1.py`

and result will save to `log.txt` and all model's confusion matrix will save to `<model_name>_confusion_matrix.png`

here is my result

<img src="./result/KNN_confusion_matrix.png" alt="KNN Confusion Matrix" width="33%"/><img src="./result/RandomForest_confusion_matrix.png" alt="Random Forest Confusion Matrix" width="33%"/><img src="./result/SVM_confusion_matrix.png" alt="SVM Confusion Matrix" width="33%"/>

and best model is SVM

Top-1 Accuracy: 0.3203  Top-3 Accuracy: 0.5411

## task2

run `python3 task2.py` for model training and predicting test set, result will save to `./pred_task2.json`

run `python3 task2_baseline.py` for baseline model training

### model structure

```text
audio -> mel spectrogram
-> CNN(2 layer FC, hidden layer shape is 128) 
-> 20 classes probability
```

### model accuracy curve

<img src="./result/accuracy_curve.png" alt="KNN Confusion Matrix" width="33%"/>

### sample for use task2 checkpoint
you can also simple use `python3 evaluate_task2.py` for generate

`confusion_matrix_task2.png`, `log_task2.txt`(validation accuracy), `pred_task2.json`(test set prediction with model)

```python
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioClassifierCNN(n_class=20).to(device)
model.load_state_dict(torch.load('./checkpoint/task2_checkpoint.pth', map_location=device))
```

### baseline structure

```text
audio -> musicFM -> embedding 
-> average pool from [time, channel] to [channel] -> MLP(2 layer FC, hidden layer shape is 128) 
-> 20 classes probability
```

### model accuracy curve

<img src="./result/accuracy_curve_baseline.png" alt="KNN Confusion Matrix" width="33%"/>

### sample for use task2 baseline checkpoint
you can also simple use `python3 evaluate_task2_baseline.py` for generate

`confusion_matrix_task2_baseline.png`, `log_task2_baseline.txt`(validation accuracy)
```python
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Classifier(2000,1024,20).to(device)

model.load_state_dict(torch.load('./checkpoint/task2_baseline_checkpoint.pth', map_location=device))
```