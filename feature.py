import os
import sys

import logging
import pandas as pd
from timeit import default_timer as timer

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class FeaturesGroup:
    
    __common_features = set()
    _group_name = "Generic group"
    
    def __init_subclass__(cls):
        cls.__common_features = set()
    
    def __init__(self, *features, group_name=None):
        if group_name is not None:
            self._group_name = group_name

        self._features = set(features)
        for feature in self._features:
            feature.add_features_group(self)

    @classmethod
    def add_common_feature(cls, feature):
        cls.__common_features.add(feature)
        
    @classmethod
    def common_features(cls):
        return cls.__common_features
        
    def add_feature(self, feature):
        self._features.add(feature)
        
    @property
    def features(self):
        return self.__common_features.union(self._features)
    
    @property
    def group_name(self):
        return self._group_name

class FeaturesFile(FeaturesGroup):
    
    _group_name = "Generic file group"
    _loaded = {}

    def __init__(self, path):
        super().__init__()
        
        self._path = os.path.abspath(path)
        if self._path not in self._loaded:
            logger.debug(f"Loading file {path} ...")
            start_loading = timer()
            
            self._loaded[self._path] = pd.read_csv(self._path, encoding = "ISO-8859-1")
            self._prepare_data()
            
            end_loading = timer()
            logger.debug(f"{path} loaded in {end_loading - start_loading:.2f}s")
        else:
            logger.debug(f"{path} already loaded")
        
    @property
    def user_grouped_df(self):
        return self._loaded[self._path]

    @property
    def path(self):
        return self._path
        
    def _prepare_data(self):
        raise NotImplementedError
    
    def _extract(self, features):
        features_columns = [
            feature.column_name
            for feature in features
        ]
        
        user_grouped_df = self.user_grouped_df
        extracted_columns = user_grouped_df[features_columns].copy()
        for feature in features:
            extracted_columns[feature.column_name] = extracted_columns[feature.column_name].apply(feature.transformation)
        
        return extracted_columns
    
    def extract(self, other_group):
        features_to_extract = self.features.intersection(other_group.features)
        return self._extract(features_to_extract)
        
class UsersFeaturesFile(FeaturesFile):
    
    _group_name = "Users group"
    
    def __init__(self, path):
        super().__init__(os.path.join(path, "users.csv"))
    
    def _prepare_data(self):
        user_grouped_df = self.user_grouped_df
        user_grouped_df.fillna('', inplace=True)
        user_grouped_df.rename(columns={"id": "user_id"}, inplace=True)
        user_grouped_df.set_index('user_id', inplace=True)
    
class FriendsFeaturesFile(FeaturesFile):
    
    _group_name = "Friends group"
    
    def __init__(self, path):
        super().__init__(os.path.join(path, "friends.csv"))
    
    def _prepare_data(self):
        self.user_grouped_df.rename(columns={"source_id": "user_id"}, inplace=True)
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)
    
class FollowersFeaturesFile(FeaturesFile):
    
    _group_name = "Followers group"
    
    def __init__(self, path):
        super().__init__(os.path.join(path, "followers.csv"))
    
    def _prepare_data(self):
        self.user_grouped_df.rename(columns={"source_id": "user_id"}, inplace=True)
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)
    
class TweetsFeaturesFile(FeaturesFile):
    
    _group_name = "Tweets group"
    
    def __init__(self, path):
        super().__init__(os.path.join(path, "tweets.csv"))
    
    def _prepare_data(self):
        user_grouped_df = self.user_grouped_df
        user_grouped_df.fillna("", inplace=True)
        user_grouped_df.set_index(["user_id", "id"], inplace=True)

class Feature:
    
    def __init__(self, name, column_name, transformation=None, features_groups=None):
        self.name = name
        self.column_name = column_name
        self.transformation = (lambda x: x) if transformation is None else transformation
        
        self._features_groups = features_groups
        
        for features_group in features_groups:
            if isinstance(features_group, FeaturesGroup):
                features_group.add_feature(self)
            elif isinstance(features_group, type):
                features_group.add_common_feature(self)

    @property
    def features_groups(self):
        return self._features_groups
    
    def add_features_group(self, features_group):
        self._features_groups.append(features_group)
        
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError
        
        return self.name == other.name

string_not_empty = lambda x: len(str(x).strip()) > 0

class_A = FeaturesGroup(group_name="Class A")
class_B = FeaturesGroup(group_name="Class B")
class_C = FeaturesGroup(group_name="Class C")
set_CC = FeaturesGroup(group_name="Set CC")

has_name = Feature("has_name", "name", transformation=string_not_empty, features_groups=[UsersFeaturesFile, class_A, set_CC])
has_image = Feature("has_image", "default_profile_image", transformation=lambda x: x != 1, features_groups=[UsersFeaturesFile, class_A, set_CC])
has_address = Feature("has_address", "location", transformation=string_not_empty, features_groups=[UsersFeaturesFile, class_A, set_CC])