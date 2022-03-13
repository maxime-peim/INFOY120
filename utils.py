import os

from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate

from dataset import Dataset

SCORING = {
    "Accuracy": "accuracy",
    "Precision": make_scorer(precision_score, pos_label="human"),
    "Recall": make_scorer(recall_score, pos_label="human"),
    "F-M": make_scorer(f1_score, pos_label="human"),
    "MCC": make_scorer(matthews_corrcoef),
    "AUC": "roc_auc",
}


def dataset_path(path, label, dataset_name):
    return os.path.join(path, label, dataset_name)


def build_paper_datasets(path):
    # building human dataset
    HUM = Dataset(
        dataset_path(path, "human", "E13"), dataset_path(path, "human", "TFP")
    )
    HUM.name = "HUM"

    # building fake dataset
    FAK = Dataset(
        dataset_path(path, "fake", "FSF"),
        dataset_path(path, "fake", "INT"),
        dataset_path(path, "fake", "TWT"),
    )
    FAK.name = "FAK"
    FAK.undersample(HUM.size)

    # building base dataset
    BAS = HUM + FAK
    BAS.name = "BAS"

    return HUM, FAK, BAS


def extract_scores(scores):
    extracted = {}
    for score_name, values in scores.items():
        if score_name.startswith("test_"):
            extracted[score_name[5:]] = mean(values)
    return extracted


def save_scores(scores, path):
    with open(path, "w") as score_out:
        for score_name, score_value in scores.items():
            score_out.write(f"{score_name}: {score_value:.8f}\n")


def get_scores(features_evaluation, labels):
    # https://machinelearningmastery.com/random-forest-ensemble-in-python/
    RF_model = RandomForestClassifier()
    full_scores = cross_validate(
        RF_model,
        features_evaluation.to_numpy(),
        labels.to_numpy(),
        scoring=SCORING,
        cv=10,
        n_jobs=-1,
        error_score="raise",
    )

    return extract_scores(full_scores)


def classify_and_save(features_evaluation, labels, path):
    scores = get_scores(features_evaluation, labels)
    save_scores(scores, path)


def get_result_from_file(path):
    if not os.path.exists(path):
        raise FileExistsError

    scores = {}
    with open(path, "r") as score_in:
        while line := score_in.readline():
            score_name, score_value = line.split(": ")
            score_value = float(score_value)
            scores[score_name] = score_value

    return scores


def normfile(filename):
    return filename.lower().replace(" ", "_")
