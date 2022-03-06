from numpy import mean, std
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

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
    n_scores = cross_val_score(RF_model, X, y, scoring='accuracy', cv=10, n_jobs=-1, error_score='raise')
    
    print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))