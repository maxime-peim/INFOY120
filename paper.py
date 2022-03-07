from numpy import mean, std
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer

import dataset as ds
import feature as ft

def build_paper_datasets():
    # building human dataset
    HUM = ds.Dataset("datasets/human/E13",
                     "datasets/human/TFP")
    HUM.name = "HUM"

    # building fake dataset
    FAK = ds.Dataset("datasets/fake/FSF",
                     "datasets/fake/INT",
                     "datasets/fake/TWT")
    FAK.name = "FAK"
    FAK.undersample(HUM.size)

    # building base dataset
    BAS = HUM + FAK
    BAS.name = "BAS"

    return HUM, FAK, BAS


if __name__ == "__main__":
    HUM, FAK, BAS = build_paper_datasets()
    
    X, y = BAS.make_classification(ft.class_A)
    
    # https://machinelearningmastery.com/random-forest-ensemble-in-python/
    RF_model = RandomForestClassifier()
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label="human"),
        "recall": make_scorer(recall_score, pos_label="human"),
        "f1": make_scorer(f1_score, pos_label="human"),
        "roc_auc": "roc_auc"
    }
    scores = cross_validate(RF_model, X, y, scoring=scoring, cv=10, n_jobs=-1, error_score='raise')
    print(scores)