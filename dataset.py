import logging
import os

import pandas as pd

import datafile as df
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InvalidDatasetFolderError(Exception):
    pass


class UserOutsideDatasetError(Exception):
    pass


class Dataset:
    """
    Ease manipulation of datasets.
    Multiple paths to csv can be passed to the constructor to build a dataset.
    The dataset's label and name depend on subdatasets from each path.
    """

    def __init__(self, *paths_to_csv):
        self._label = None
        self._names = set()
        self._full_users = None
        self._users = None
        self._datafiles = {}
        self._undersampled = False

        self._set_from_paths(paths_to_csv)

    @property
    def size(self):
        if self._users is None:
            return 0
        return len(self._users.index)

    @property
    def undersampled(self):
        return self._undersampled

    @property
    def users(self):
        if self._users is None:
            return pd.DataFrame()
        return self._users

    @users.setter
    def users(self, users):
        if not users.index.isin(self._full_users.index).all():
            raise UserOutsideDatasetError

        if len(users.index) == len(self._full_users.index):
            self._undersampled = False
            self._users = self._full_users
        else:
            self._undersampled = True
            self._users = users

    @property
    def datafiles(self):
        return self._datafiles

    @datafiles.setter
    def datafiles(self, datafiles):
        paths = list(datafiles.keys())
        self._set_from_paths(paths)

    @property
    def name(self):
        return " U ".join(self._names)

    @name.setter
    def name(self, name):
        self._names.clear()
        self._names.add(name)

    @property
    def paths_to_csv(self):
        return list(self._datafiles.keys())

    @paths_to_csv.setter
    def paths_to_csv(self, paths):
        self._set_from_paths(paths)

    def reset_full_users(self):
        self._users = self._full_users

    def _set_from_paths(self, paths):
        self._datafiles.clear()

        users_dfs = []
        for path in paths:
            path = os.path.normpath(path)
            users_df, label, name = self.extract_users_from_path(path)

            users_dfs.append(users_df)

            if self._label is None:
                self._label = label
            elif self._label != label:
                self._label = "mixed"

            self._names.add(name)

            logger.debug(f"Loading features files from {path}...")
            self._datafiles[path] = (
                df.MinimalDatafiles(path) if not utils.TESTING else None
            )

        self._undersampled = False
        self._full_users = (
            pd.concat(users_dfs) if len(users_dfs) > 0 else pd.DataFrame()
        )
        self._users = self._full_users

    @staticmethod
    def parse_path(path):
        path = os.path.normpath(path)
        path_parts = path.split(os.sep)

        if len(path_parts) >= 2:
            return path_parts[-2], path_parts[-1]

        raise InvalidDatasetFolderError(
            f"Path {path} is not a valid path to CSV files."
        )

    @staticmethod
    def extract_users_from_path(path):
        users_path = os.path.join(path, "users.csv")
        mandatory_files = [
            users_path,
            os.path.join(path, "followers.csv"),
            os.path.join(path, "friends.csv"),
            os.path.join(path, "tweets.csv"),
        ]

        for mandatory_file in mandatory_files:
            if not os.path.exists(mandatory_file):
                raise InvalidDatasetFolderError(f"{mandatory_file} is missing.")

        label, dataset_name = Dataset.parse_path(path)

        users_df = pd.read_csv(users_path, usecols=["id"])
        users_df["label"] = label
        users_df["dataset"] = dataset_name
        users_df.rename(columns={"id": "user_id"}, inplace=True)
        users_df.set_index("user_id", inplace=True)

        return users_df, label, dataset_name

    def copy(self):
        return type(self)(*self.paths_to_csv)

    def undersample(self, num_points):
        self._users = self._full_users.sample(n=num_points)
        self._undersampled = True

    def make_classification(self, features_group):
        users_features = []

        for datafiles in self._datafiles.values():
            sub_users_features = datafiles.extract(features_group)
            users_features.append(sub_users_features)

        X = pd.concat(users_features)
        X = X.reindex(self._users.index)
        return X.to_numpy(), self._users["label"].to_numpy(copy=True)

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        # save the users if undersampled before modifying
        undersampled_users = pd.concat((self.users, other.users))

        self.paths_to_csv = self.paths_to_csv + other.paths_to_csv
        self.users = undersampled_users

        return self

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        # save the users if undersampled before modifying
        undersampled_users = pd.concat((self.users, other.users))

        paths = self.paths_to_csv + other.paths_to_csv
        new_dataset = type(self)(*paths)
        new_dataset.users = undersampled_users

        return new_dataset

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.users.equals(other.users)

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.users.index.isin(other.users.index).all()

    def __gt__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return other.users.index.isin(self.users.index).all()

    def __le__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.__eq__(other) or self.__lt__(other)

    def __ge__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.__eq__(other) or self.__gt__(other)

    def __str__(self):
        return f"Dataset '{self.name}', labeled {self._label}, with {self.size} users."
