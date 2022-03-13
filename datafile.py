from __future__ import annotations

import logging
import os
from timeit import default_timer as timer

import pandas as pd

import feature as ft

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Datafile:
    """Ease loading of datafile from a dataset.

    Raises:
        NotImplementedError: each datafile type must implement _prepare_data depending on its structure.
    """

    _loaded = {}
    _classes = {}

    def __init_subclass__(cls, /, prefix: str, **kwargs):
        """Register each datafile type based on its prefix.

        Args:
            prefix (str): the prefix for the datafile type.
        """
        super().__init_subclass__(**kwargs)
        cls.__prefix = prefix
        cls._classes[prefix] = cls

    def __init__(self, path: str):
        """Loads the datafile in memory and prepare the dataframe.

        Args:
            path (str): path to the datafile.
        """
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
    def user_grouped_df(self) -> pd.Dataframe:
        """Returns the dataframe associated to the datafile.

        Returns:
            pd.Dataframe: dataframe grouped by user id
        """
        return self._loaded[self._path]

    @property
    def path(self) -> str:
        """Returns the path to the datafile.

        Returns:
            str: path to the datafile.
        """
        return self._path

    @classmethod
    @property
    def prefix(cls) -> str:
        """Returns the prefix for the datafile.

        Returns:
            str: prefix for the datafile.
        """
        return cls.__prefix

    @classmethod
    def get_class(cls, prefix: str) -> type:
        """Returns the class associated with a prefix.

        Args:
            prefix (str): prefix of a class of datafile.

        Returns:
            type: the class for the datafile.
        """
        return cls._classes.get(prefix, cls)

    def _prepare_data(self):
        """Should prepare the dataframe associated with the datafile, and group the dataframe by user id.

        Raises:
            NotImplementedError: each subclass should implement this method.
        """
        raise NotImplementedError

    def extract(self, columns_names: list[str]) -> pd.DataFrame:
        """Returns the columns asked from the dataframe.

        Args:
            columns_names (list[str]): a list of columns names.

        Returns:
            pd.DataFrame: the resulting columns extracted.
        """
        return self.user_grouped_df[columns_names].rename(
            columns={name: f"{self.prefix}_{name}" for name in columns_names}
        )


class UsersDatafile(Datafile, prefix="users"):
    """Datafile for users.csv files."""

    def _prepare_data(self):
        user_grouped_df = self.user_grouped_df
        user_grouped_df.fillna("", inplace=True)
        user_grouped_df.rename(columns={"id": "user_id"}, inplace=True)
        user_grouped_df.set_index("user_id", inplace=True)


class FriendsDatafile(Datafile, prefix="friends"):
    """Datafile for friends.csv files."""

    def _prepare_data(self):
        """We assumed that the file friends.csv contains a list of link as A follows B
        (it can be confirmed by comparing the number of friends in users.csv).
        Hence to build a list of friends for each user, the source_id column is taken as
        user_id index.
        """
        self.user_grouped_df.rename(
            columns={"source_id": "user_id", "target_id": "friends_id"}, inplace=True
        )
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)


class FollowersDatafile(Datafile, prefix="followers"):
    """Datafile for followers.csv files."""

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
    """Datafile for tweets.csv files."""

    def _prepare_data(self):
        self.user_grouped_df.fillna("", inplace=True)
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)


class MinimalDatafiles:
    """Class that contains datafiles object for a minimal dataset."""

    def __init__(self, path: str):
        self.path = path

        self.users = UsersDatafile(path)
        self.friends = FriendsDatafile(path)
        self.followers = FollowersDatafile(path)
        self.tweets = TweetsDatafile(path)

        self.files = {
            self.users.prefix: self.users,
            self.friends.prefix: self.friends,
            self.followers.prefix: self.followers,
            self.tweets.prefix: self.tweets,
        }

    def extract(self, feature: ft.Feature) -> pd.DataFrame:
        """Extract the needed columns over all datafile for a feature.

        Args:
            feature (ft.Feature): the feature for which we want to extract the needed columns.

        Returns:
            pd.DataFrame: the resulting dataframe with all the needed columns.
        """
        users_index = self.users.user_grouped_df.index
        extracted_columns = []
        for prefix, datafile in self.files.items():
            columns_names = feature.needed_columns(prefix)
            if len(columns_names) > 0:
                columns = datafile.extract(columns_names)
                if not columns.empty:
                    extracted_columns.append(columns)

        return pd.concat(extracted_columns, axis=1).reindex(index=users_index)
