import logging
import sys

import utils
import feature as ft

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout)

    HUM, FAK, BAS = utils.build_paper_datasets()
    features_evaluation, labels = BAS.evaluate_features(ft.all_features)

    evaluations = [
        (ft.class_A, "results/paper/class_A_alone.txt"),
        (ft.class_B, "results/paper/class_B_alone.txt"),
        (ft.class_C, "results/paper/class_C_alone.txt"),
        (ft.class_A | ft.class_B, "results/paper/class_AB.txt"),
        (ft.all_features, "results/paper/all_features.txt"),
        (ft.set_Y, "results/paper/set_yang.txt"),
        (ft.set_S, "results/paper/set_stringhini.txt"),
    ]

    for features, path in evaluations:
        logger.info(f"Evaluating {features.name} over all datasets.")
        utils.classify_and_save(features_evaluation[features.names], labels, path)
