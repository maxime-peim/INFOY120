import os
import random

import pandas as pd
import dataclasses as dc
from enum import Enum, auto


@dc.dataclass
class Users:
    id: int
    label: str
    dataset_name: str

    def __hash__(self):
        return hash((self.id, self.label, self.dataset_name))

    def __eq__(self, other):
        if not isinstance(other, Users):
            return NotImplemented
        return self.id == other.id

class InvalidDatasetFolderError(Exception):
    pass

class Dataset:
    """
        Ease manipulation of datasets.
        Multiple paths to csv can be passed to the constructor to build a dataset.
        The dataset's label and name depend on subdatasets from each path.
    """

    def __init__(self, *paths_to_csv):
        self.name = None
        self._label = None
        self._paths_to_csv = set(paths_to_csv)
        self._users = set()
        
        self._set_users()

    @property
    def users(self):
        return self._users

    @property
    def paths_to_csv(self):
        return self._paths_to_csv

    @property
    def size(self):
        return len(self._users)

    @paths_to_csv.setter
    def paths_to_csv(self, *paths):
        self._paths_to_csv = set(paths)
        self._set_users()

    def _set_users(self):
        names = []
        for path in self._paths_to_csv:
            users, label, name = self.extract_users_from_path(path)

            self._users |= users

            if self._label is None:
                self._label = label
            elif self._label != label:
                self._label = "mixed"

            names.append(name)
        
        self.name = " U ".join(names)
            
    @staticmethod
    def parse_path(path):
        path = os.path.normpath(path)
        path_parts = path.split(os.sep)

        if len(path_parts) >= 3 and path_parts[-3] == "datasets":
            return path_parts[-2], path_parts[-1]

        raise InvalidDatasetFolderError(f"Path {path} is not a valid path to CSV files.")

    @staticmethod
    def extract_users_from_path(path):
        users_path = os.path.join(path, "users.csv")
        mandatory_files = [
            users_path,
            os.path.join(path, "followers.csv"),
            os.path.join(path, "friends.csv"),
            os.path.join(path, "tweets.csv")
        ]

        for mandatory_file in mandatory_files:
            if not os.path.exists(mandatory_file):
                raise InvalidDatasetFolderError(f"{mandatory_file} is missing.")
        
        users_id = pd.read_csv(users_path, usecols=["id"])
        label, name = Dataset.parse_path(path)
        users = [
            Users(row.id, label, name)
            for index, row in users_id.iterrows()
        ]

        return set(users), label, name

    def copy(self):
        return Dataset(*self._paths_to_csv)

    def __add__(self, other):
        if not isinstance(other, Dataset):
            raise NotImplemented

        if self.paths_to_csv == other.paths_to_csv:
            return self.copy()
        
        paths_to_csv = self.paths_to_csv.union(other.paths_to_csv)

        return Dataset(*paths_to_csv)

    def inplace_undersample(self, num_points):
        self._users = random.sample(self._users, num_points)
        self._paths_to_csv = set(
            os.path.join("datasets", user.label, user.dataset_name)
            for user in self._users
        )

    def undersample(self, num_points):
        new_dataset = self.copy()
        new_dataset.inplace_undersample(num_points)
        
        return new_dataset
    
    def __str__(self):
        return f"Dataset '{self.name}', labeled {self._label}, with {len(self._users)} users."