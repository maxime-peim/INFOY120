import os
import random

import pandas as pd
import dataclasses as dc

@dc.dataclass
class Users:
    id: int
    label: str
    dataset_name: str

    def __hash__(self):
        return hash((self.id, self.label, self.dataset_name))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplementedError
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
        self._label = None
        self._names = set()
        self._paths_to_csv = set(paths_to_csv)
        self._users = set()
        
        self._set_users_from_paths()

    @property
    def size(self):
        return len(self._users)

    @property
    def users(self):
        return self._users

    @property
    def name(self):
        return " U ".join(self._names)

    @name.setter
    def name(self, name):
        self._names.clear()
        self._names.add(name)

    @users.setter
    def users(self, users):
        self._users = set(users)
        self._paths_to_csv.clear()
        self._names.clear()

        self._label = None

        for user in users:
            self._paths_to_csv.add(
                os.path.join("datasets", user.label, user.dataset_name))
            
            if self._label is None:
                self._label = user.label
            elif self._label != user.label:
                self._label = "mixed"
            
            self._names.add(user.dataset_name)

    @property
    def paths_to_csv(self):
        return self._paths_to_csv

    @paths_to_csv.setter
    def paths_to_csv(self, *paths):
        self._paths_to_csv = set(paths)
        self._set_users_from_paths()

    @classmethod
    def from_users(cls, users):
        new_dataset = cls()
        new_dataset.users = users

        return new_dataset

    def _set_users_from_paths(self):
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
        return type(self)(*self._paths_to_csv)

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

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        self.users = self.users.union(other.users)

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        users = self.users.union(other.users)

        return type(self).from_users(users)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
           raise NotImplementedError

        return self.users == other.users

    def __lt__(self, other):
        if not isinstance(other, type(self)):
           raise NotImplementedError

        return self.users.issubset(other.users)

    def __gt__(self, other):
        if not isinstance(other, type(self)):
           raise NotImplementedError

        return other.users.issubset(self.users)

    def __le__(self, other):
        if not isinstance(other, type(self)):
           raise NotImplementedError

        return self.__eq__(other) or self.__lt__(other)

    def __ge__(self, other):
        if not isinstance(other, type(self)):
           raise NotImplementedError

        return self.__eq__(other) or self.__gt__(other)
    
    def __str__(self):
        return f"Dataset '{self.name}', labeled {self._label}, with {len(self._users)} users."