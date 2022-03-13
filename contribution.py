import argparse
import glob
import logging
import os
import re
import sys

from numpy import mean

import feature as ft
import utils

logger = logging.getLogger(__name__)
SCORES_FOLDER_TEMPLATE = "results/contribution/{group}"
SCORES_FILE_TEMPLATE = os.path.join(SCORES_FOLDER_TEMPLATE, "{features_names}.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("path", type="str", help="Path to datasets")
    parser.add_argument("-v", dest="verbose", action="store_true", help="Verbose")
    parser.add_argument("-c", "--custom", nargs="+", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="Custom group")
    parser.add_argument("-s", "--single-eval", action="store_true")
    parser.add_argument("-b", "--best", action="store_true")
    parser.add_argument("-m", "--min", action="store_true")

    args = parser.parse_args()

    custom_features_names = args.custom
    available_custom_features_names = []
    best = args.best
    single_eval = args.single_eval
    custom_name = args.name
    min_or_mean = min if args.min else mean

    verbose_level = logging.INFO if args.verbose else logging.NOTSET
    logger.setLevel(verbose_level)
    logging.captureWarnings(True)
    logging.basicConfig(stream=sys.stdout)

    HUM, FAK, BAS = utils.build_paper_datasets(args.path)
    if single_eval or best:
        features_evaluation, labels = BAS.evaluate_features(ft.all_features)
    elif custom_features_names is not None:
        custom_features = ft.FeaturesGroup()
        for feature_name in custom_features_names:
            if feature_name in ft.Feature.registered:
                available_custom_features_names.append(feature_name)
                custom_features.add(ft.Feature.registered[feature_name])
        features_evaluation, labels = BAS.evaluate_features(custom_features)

    if single_eval or best:
        single_scores = {}
        for feature in ft.all_features:
            logger.info(f"Getting scores for {feature} alone.")
            scores = utils.get_scores(features_evaluation[[feature.name]], labels)
            single_scores[feature.name] = scores

        if single_eval:
            for feature_name, scores in single_scores.items():
                logger.info(f"Saving scores for {feature_name}")
                utils.save_scores(
                    scores,
                    SCORES_FILE_TEMPLATE.format(
                        group="single", features_names=f"feature_{feature_name}"
                    ),
                )

        if best:
            BEST_FOLDER = SCORES_FOLDER_TEMPLATE.format(group="best/min")
            BEST_FILE_PATTERN = re.compile(r"run_(\d+).txt")
            files = glob.glob(os.path.join(BEST_FOLDER, "*.txt"))
            next_file = (
                1
                if len(files) == 0
                else max(
                    int(BEST_FILE_PATTERN.match(os.path.basename(file_name)).group(1))
                    for file_name in files
                )
                + 1
            )

            best_features = []
            best_scores = None
            best_mean_score = 0

            sorted_features = sorted(
                single_scores.keys(),
                key=lambda name: min_or_mean(list(single_scores[name].values())),
                reverse=True,
            )
            for feature_name in sorted_features:
                best_features.append(feature_name)
                logger.info(f"Trying a set of features of size {len(best_features)}")
                scores = utils.get_scores(features_evaluation[best_features], labels)
                mean_score = min_or_mean(list(scores.values()))

                if mean_score > best_mean_score:
                    best_scores = scores
                    best_mean_score = mean_score
                else:
                    best_features.pop()

            utils.save_scores(
                best_scores, os.path.join(BEST_FOLDER, f"run_{next_file}.txt")
            )
            print(best_features)

    if len(available_custom_features_names):
        logger.info(f"Getting scores for {custom_name}.")
        utils.classify_and_save(
            features_evaluation[available_custom_features_names],
            labels,
            SCORES_FILE_TEMPLATE.format(
                group="custom", features_names=utils.normfile(custom_name)
            ),
        )
