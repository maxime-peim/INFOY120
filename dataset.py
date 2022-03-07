import os
import random

import pandas as pd

import utils
import feature as ft

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
        self._features_files = {}
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
    def features_files(self):
        return self._features_files
    
    @features_files.setter
    def features_files(self, features_files):
        paths = list(features_files.keys())
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
        return list(self._features_files.keys())

    @paths_to_csv.setter
    def paths_to_csv(self, paths):
        self._set_from_paths(paths)
        
    def reset_full_users(self):
        self._users = self._full_users

    def _set_from_paths(self, paths):
        self._features_files.clear()
        
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
            
            self._features_files[path] = (
                ft.UsersFeaturesFile(path),
                ft.FriendsFeaturesFile(path),
                ft.FollowersFeaturesFile(path),
                ft.TweetsFeaturesFile(path)
            ) if not utils.TESTING else ()

        self._undersampled = False
        self._full_users= pd.concat(users_dfs) if len(users_dfs) > 0 else pd.DataFrame()
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
        
        label, name = Dataset.parse_path(path)
        
        users_df = pd.read_csv(users_path, usecols=["id", "dataset"])
        users_df["label"] = label
        users_df.rename(columns={"id": "user_id"}, inplace=True)
        users_df.set_index("user_id", inplace=True)

        return users_df, label, name

    def copy(self):
        return type(self)(*self.paths_to_csv)

    def undersample(self, num_points):
        self._users = self._full_users.sample(n=num_points)
        self._undersampled = True
    
    def make_classification(self, features_group):
        users_features = []
        
        for features_files in self._features_files.values():
            sub_users_features = []
            for features_file in features_files:
                id_intersection = self._users.index.intersection(features_file.user_grouped_df.index)
                extracted = features_file.extract(features_group)
                if not extracted.empty:
                    sub_users_features.append(extracted.loc[id_intersection])

            users_features.append(pd.concat(sub_users_features, axis=1))
            
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
