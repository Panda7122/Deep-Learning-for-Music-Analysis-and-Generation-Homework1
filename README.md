# Deep-Learning-for-Music-Analysis-and-Generation-Homework1 README

## 41147009S 陳炫佑

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

![KNN Confusion Matrix](./result/KNN_confusion_matrix.png)
![Random Forest Confusion Matrix](./result/RandomForest_confusion_matrix.png)
![SVN Confusion Matrix](./result/SVM_confusion_matrix.png)
