import logging
import os
from dataclasses import dataclass, field
from timeit import default_timer as timer

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Datafile:

    _loaded = {}
    _classes = {}

    def __init_subclass__(cls, /, prefix, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__prefix = prefix
        cls._classes[prefix] = cls

    def __init__(self, path):
        path = os.path.join(os.path.normpath(path), f"{self.__prefix}.csv")
        self._path = os.path.abspath(path)
        if self._path not in self._loaded:
            start_loading = timer()

            self._loaded[self._path] = pd.read_csv(self._path, encoding="ISO-8859-1")
            self._prepare_data()

            end_loading = timer()
            logger.debug(
                f"{os.path.basename(path)} loaded in {end_loading - start_loading:.2f}s"
            )
        else:
            logger.debug(f"{os.path.basename(path)} already loaded")

    @property
    def user_grouped_df(self):
        return self._loaded[self._path]

    @property
    def path(self):
        return self._path

    @classmethod
    @property
    def prefix(cls):
        return cls.__prefix

    @classmethod
    def get_class(cls, prefix):
        return cls._classes.get(prefix, cls)

    def _prepare_data(self):
        raise NotImplementedError

    def extract(self, columns_names):
        return self.user_grouped_df[columns_names].rename(
            columns={name: f"{self.prefix}_{name}" for name in columns_names}
        )


class UsersDatafile(Datafile, prefix="users"):
    def _prepare_data(self):
        user_grouped_df = self.user_grouped_df
        user_grouped_df.fillna("", inplace=True)
        user_grouped_df.rename(columns={"id": "user_id"}, inplace=True)
        user_grouped_df.set_index("user_id", inplace=True)


class FriendsDatafile(Datafile, prefix="friends"):
    def _prepare_data(self):
        """We assumed that the file friends.csv contains a list of link as A follows B
        (it can be confirmed by comparing the number of followers in users.csv).
        Hence to build a list of friends for each user, the source_id column is taken as
        user_id index.
        """
        self.user_grouped_df.rename(
            columns={"source_id": "user_id", "target_id": "friends_id"}, inplace=True
        )
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)


class FollowersDatafile(Datafile, prefix="followers"):
    def _prepare_data(self):
        """We assumed that the file followers.csv contains a list of link as A follows B
        (it can be confirmed by comparing the number of followers in users.csv).
        Hence to build a list of followers for each user, the target_id column is taken as
        user_id index.
        """
        self.user_grouped_df.rename(
            columns={"target_id": "user_id", "source_id": "followers_id"}, inplace=True
        )
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)


class TweetsDatafile(Datafile, prefix="tweets"):
    def _prepare_data(self):
        self.user_grouped_df.fillna("", inplace=True)
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)


@dataclass
class MinimalDatafiles:
    path: str
    files: dict = field(init=False)
    users: UsersDatafile = field(init=False)
    friends: FriendsDatafile = field(init=False)
    followers: FollowersDatafile = field(init=False)
    tweets: TweetsDatafile = field(init=False)

    def __post_init__(self):
        if self.path is None:
            raise ValueError("Cannot load datafiles without path.")

        self.users = UsersDatafile(self.path)
        self.friends = FriendsDatafile(self.path)
        self.followers = FollowersDatafile(self.path)
        self.tweets = TweetsDatafile(self.path)

        self.files = {
            self.users.prefix: self.users,
            self.friends.prefix: self.friends,
            self.followers.prefix: self.followers,
            self.tweets.prefix: self.tweets,
        }

    def extract(self, feature):
        users_index = self.users.user_grouped_df.index
        extracted_columns = []
        for prefix, datafile in self.files.items():
            columns_names = feature.needed_columns(prefix)
            if len(columns_names) > 0:
                columns = datafile.extract(columns_names)
                if not columns.empty:
                    extracted_columns.append(columns)

        return pd.concat(extracted_columns, axis=1).reindex(index=users_index)
