from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Union

import pandas as pd

import datafile as df
import feature as ft

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InvalidDatasetFolderError(Exception):
    pass


class UserOutsideDatasetError(Exception):
    pass


class Dataset:
    """A class that eases manipulation of datasets and their manipulation."""

    def __init__(self, *paths_to_csv):
        """Constructor for a dataset.
        It takes a list of path to minimal datasets as E13 or TFP.
        """
        self._label = None
        self._names = set()
        self._full_users = None
        self._users = None
        self._datafiles = {}
        self._undersampled = False

        self._set_from_paths(paths_to_csv)

    @property
    def size(self) -> int:
        """
        Returns:
            int: the number of users in the dataset.
        """
        if self._users is None:
            return 0
        return len(self._users.index)

    @property
    def undersampled(self) -> bool:
        """
        Returns:
            bool: True if the dataset has been undersampled.
        """
        return self._undersampled

    @property
    def users(self) -> pd.DataFrame:
        """
        Returns:
            pd.DataFrame: a dataframe containing the users of the dataset.
        """
        if self._users is None:
            return pd.DataFrame()
        return self._users

    @users.setter
    def users(self, users: pd.DataFrame):
        """Checks whether users are in the dataset and
        changes the current users of the dataset.

        Args:
            users (pd.DataFrame): a dataframe of users to be used.

        Raises:
            UserOutsideDatasetError: raised when some users passed
            in argument are not contained in the dataset.
        """
        if not users.index.isin(self._full_users.index).all():
            raise UserOutsideDatasetError

        if len(users.index) == len(self._full_users.index):
            self._undersampled = False
            self._users = self._full_users
        else:
            self._undersampled = True
            self._users = users

    @property
    def datafiles(self) -> dict[df.Datafile]:
        """
        Returns:
            dict[df.Datafile]: a dictionnary keyed by path to the files forming the dataset.
        """
        return self._datafiles

    @property
    def name(self) -> str:
        """Build the name of the dataset based on subdatasets.

        Returns:
            str: the concatenated name of the subdatasets.
        """
        return " U ".join(self._names)

    @name.setter
    def name(self, name: str):
        """Set the name of the dataset.

        Args:
            name (str): the wanted name for the dataset.
        """
        self._names.clear()
        self._names.add(name)

    @property
    def paths_to_csv(self) -> list[str]:
        """
        Returns:
            list[str]: the list of paths to datafiles.
        """
        return list(self._datafiles.keys())

    @paths_to_csv.setter
    def paths_to_csv(self, paths: list[str]):
        """Change the dataset files based on paths to datafiles.

        Args:
            paths (list[str]): list of paths to datafiles.
        """
        self._set_from_paths(paths)

    def reset_full_users(self):
        """If the dataset has been undersampled,
        one could want to restore the full users dataframe.
        """
        self._undersampled = False
        self._users = self._full_users

    def _set_from_paths(self, paths: list[str]):
        """Sets the users dataframe, the datafiles dictionnary and names of the dataset,
        based on paths to datafiles.

        Args:
            paths (list[str]): paths to datafiles.
        """
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
            self._datafiles[path] = df.MinimalDatafiles(path)

        self._undersampled = False
        self._full_users = (
            pd.concat(users_dfs) if len(users_dfs) > 0 else pd.DataFrame()
        )
        self._users = self._full_users

    @staticmethod
    def parse_path(path: str) -> tuple[str, str]:
        """Takes a path to datafiles, formatted as "/.../<label>/<dataset name>/",
        and return label and dataset name.

        Args:
            path (str): formatted path to datafiles.

        Raises:
            InvalidDatasetFolderError: raised if the path is not well formatted.

        Returns:
            tuple[str, str]: the label and the dataset name.
        """
        path = os.path.normpath(path)
        path_parts = path.split(os.sep)

        if len(path_parts) >= 2:
            return path_parts[-2], path_parts[-1]

        raise InvalidDatasetFolderError(
            f"Path {path} is not a valid path to CSV files."
        )

    @staticmethod
    def extract_users_from_path(path: str) -> tuple[pd.DataFrame, str, str]:
        """From a path to datafiles, extract a dataframe of users from those datafiles, the label and the subdataset name.
        Checks also if all mandatory files are present in the folder.

        Args:
            path (str): path to datafiles.

        Raises:
            InvalidDatasetFolderError: raised if missing mandatory files.

        Returns:
            tuple[pd.DataFrame, str, str]: the users dataframe from the datafiles, the label and the subdataset name.
        """
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

    def copy(self) -> Dataset:
        """Rebuild a dataset from paths.

        Returns:
            Dataset: a copy of the current dataset.
        """
        return type(self)(*self.paths_to_csv)

    def undersample(self, num_users: int):
        """In-place undersampling of the dataset.
        If the number of users asked is greater of equal to the numbers of users
        in the dataset, the users dataframe is reset to the full users dataframe.

        Args:
            num_users (int): the number of users to randomly extract from the dataset.
        """
        if num_users >= self.size:
            self.reset_full_users()
        else:
            self._users = self._full_users.sample(n=num_users)
            self._undersampled = True

    def make_classification(
        self, features: Union[ft.Feature, ft.FeaturesGroup]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from the dataset.
        It returns the features evaluation for each user of the dataset, and the label associated.

        Args:
            features (Union[ft.Feature, ft.FeaturesGroup]): a group of features to extract.

        Returns:
            tuple[np.ndarray, np.ndarray]: the first numpy ndarray contains the different features evaluation, and the second the label for each user.
        """
        users_features = []

        if isinstance(features, ft.Feature):
            features = ft.FeaturesGroup(iterable=[features])

        for datafiles in self._datafiles.values():
            sub_users_features = datafiles.extract(features)
            users_features.append(sub_users_features)

        X = pd.concat(users_features)
        X = X.reindex(self._users.index)
        return X.to_numpy(), self._users["label"].to_numpy(copy=True)

    def __iadd__(self, other: Any) -> Dataset:
        """In-place union of two datasets.

        Args:
            other (Any): if other is an instance of Dataset, it is used to add files to the current dataset.

        Raises:
            NotImplementedError: raised if other is not an instance on Dataset.

        Returns:
            Dataset: return the current dataset.
        """
        if not isinstance(other, type(self)):
            raise NotImplementedError

        # save the users if undersampled before modifying
        undersampled_users = pd.concat((self.users, other.users))

        self.paths_to_csv = self.paths_to_csv + other.paths_to_csv
        self.users = undersampled_users

        return self

    def __add__(self, other: Any) -> Dataset:
        """Union of two datasets.

        Args:
            other (Any): if other is an instance of Dataset, it is used to add files to the current dataset.

        Raises:
            NotImplementedError: raised if other is not an instance on Dataset.

        Returns:
            Dataset: return a dataset as the union of the current and other.
        """
        if not isinstance(other, type(self)):
            raise NotImplementedError

        # save the users if undersampled before modifying
        undersampled_users = pd.concat((self.users, other.users))

        paths = self.paths_to_csv + other.paths_to_csv
        new_dataset = type(self)(*paths)
        new_dataset.users = undersampled_users

        return new_dataset

    def __eq__(self, other: Any) -> bool:
        """Checks equality of two datasets based on their users.

        Args:
            other (Any): an other object to be compared.

        Returns:
            bool: True if the user are the same in both datasets.
        """
        if not isinstance(other, type(self)):
            return False

        return self.users.equals(other.users)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.users.index.isin(other.users.index).all()

    def __gt__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return other.users.index.isin(self.users.index).all()

    def __le__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.__eq__(other) or self.__lt__(other)

    def __ge__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False

        return self.__eq__(other) or self.__gt__(other)

    def __str__(self) -> str:
        return (
            f"Dataset '{self.name}', labeled {self._label}, with {self.size} users from datafiles in\n"
            + "\n".join(self.paths_to_csv)
        )
