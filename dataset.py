import os
import random

import pandas as pd

import utils
import feature as ft
from user import User

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
        self._full_users = set()
        self._users = set()
        self._features_files = {}
        self._undersampled = False
        
        self._set_from_paths(paths_to_csv)

    @property
    def size(self):
        return len(self._users)

    @property
    def undersampled(self):
        return self._undersampled

    @property
    def users(self):
        return self._users
    
    @users.setter
    def users(self, users):
        if not users.issubset(self._full_users):
            raise UserOutsideDatasetError
        
        if len(users) == self._full_users:
            self._undersampled = False
            self._users = self._full_users
        else:
            self._undersampled = True
            self._users = users

    @property
    def name(self):
        return " U ".join(self._names)

    @name.setter
    def name(self, name):
        self._names.clear()
        self._names.add(name)
        
    @property
    def paths_to_csv(self):
        return list(self._features_files.keys())

    @paths_to_csv.setter
    def paths_to_csv(self, paths):
        self._set_from_paths(paths)
        
    def reset_full_users(self):
        self._users = self._full_users

    def _set_from_paths(self, paths):
        self._full_users.clear()
        self._features_files.clear()
        
        for path in paths:
            path = os.path.normpath(path)
            users, label, name = self.extract_users_from_path(path)

            self._full_users |= users

            if self._label is None:
                self._label = label
            elif self._label != label:
                self._label = "mixed"

            self._names.add(name)
            self._features_files[path] = (
                ft.UsersFeaturesFile(path),
                ft.FriendsFeaturesFile(path),
                ft.FollowersFeaturesFile(path),
                ft.TweetsFeaturesFile(path)
            ) if not utils.TESTING else ()

        self._undersampled = False
        self._users = self._full_users
            
    @staticmethod
    def parse_path(path):
        path = os.path.normpath(path)
        path_parts = path.split(os.sep)

        if len(path_parts) >= 2:
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
            User(row.id, label, name)
            for index, row in users_id.iterrows()
        ]

        return set(users), label, name

    def copy(self):
        return type(self)(*self.paths_to_csv)

    def undersample(self, num_points):
        self._users = random.sample(self._users, num_points)
        self._undersampled = True
    
    def make_classification(self, features_group):
        pass

    def __iadd__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        self.paths_to_csv = self.paths_to_csv + other.paths_to_csv
        self.users = self.users.union(other.users)
        
        return self

    def __add__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        paths = self.paths_to_csv + other.paths_to_csv
        new_dataset = type(self)(*paths)
        
        new_dataset.users = self.users.union(other.users)

        return new_dataset

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
