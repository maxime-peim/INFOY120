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

    def __init__(self, path, *features, group_name=None):
        super().__init__(*features, group_name=group_name)
        
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
        user_grouped_df = self.user_grouped_df
        
        features_values = pd.DataFrame()
        for feature in features:
            features_values[feature.name] = user_grouped_df[feature.column_names].apply(feature.transformation, axis=1)
        
        return features_values
    
    def extract(self, other_group):
        features_to_extract = self.features.intersection(other_group.features)
        return self._extract(features_to_extract)
        
class UsersFeaturesFile(FeaturesFile):
    
    _group_name = "Users group"
    
    def __init__(self, path, *features, group_name=None):
        super().__init__(os.path.join(path, "users.csv"), *features, group_name=group_name)
    
    def _prepare_data(self):
        user_grouped_df = self.user_grouped_df
        user_grouped_df.fillna('', inplace=True)
        user_grouped_df.rename(columns={"id": "user_id"}, inplace=True)
        user_grouped_df.set_index('user_id', inplace=True)
    
class FriendsFeaturesFile(FeaturesFile):
    
    _group_name = "Friends group"
    
    def __init__(self, path, *features, group_name=None):
        super().__init__(os.path.join(path, "friends.csv"), *features, group_name=group_name)
    
    def _prepare_data(self):
        self.user_grouped_df.rename(columns={"source_id": "user_id"}, inplace=True)
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)
    
class FollowersFeaturesFile(FeaturesFile):
    
    _group_name = "Followers group"
    
    def __init__(self, path, *features, group_name=None):
        super().__init__(os.path.join(path, "followers.csv"), *features, group_name=group_name)
    
    def _prepare_data(self):
        self.user_grouped_df.rename(columns={"source_id": "user_id"}, inplace=True)
        self._loaded[self._path] = self.user_grouped_df.groupby("user_id").agg(list)
    
class TweetsFeaturesFile(FeaturesFile):
    
    _group_name = "Tweets group"
    
    def __init__(self, path, *features, group_name=None):
        super().__init__(os.path.join(path, "tweets.csv"), *features, group_name=group_name)
    
    def _prepare_data(self):
        user_grouped_df = self.user_grouped_df
        user_grouped_df.fillna("", inplace=True)
        user_grouped_df.set_index(["user_id", "id"], inplace=True)

class Feature:
    
    def __init__(self, name, column_names, transformation, features_groups=None):
        self.name = name
        self.column_names = [column_names] if isinstance(column_names, str) else column_names
        self.transformation = self._transformation_proxy(transformation)
        
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
        
    def _transformation_proxy(self, transformation):
        def transformation_wrapper(x):
            args = (
                x[column_name]
                for column_name in self.column_names
            )
            return transformation(*args)
        return transformation_wrapper
        
    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError
        
        return self.name == other.name

string_empty = lambda x: len(str(x).strip()) == 0
string_not_empty = lambda x: len(str(x).strip()) > 0
identity = lambda x: x

class_A = FeaturesGroup(group_name="Class A")
class_B = FeaturesGroup(group_name="Class B")
class_C = FeaturesGroup(group_name="Class C")
set_CC = FeaturesGroup(group_name="Set CC")

has_name = Feature("has_name", "name", string_not_empty, features_groups=[UsersFeaturesFile, class_A])
has_image = Feature("has_image", "default_profile_image", lambda x: x != 1, features_groups=[UsersFeaturesFile, class_A])
has_address = Feature("has_address", "location", string_not_empty, features_groups=[UsersFeaturesFile, class_A])
has_biography = Feature("has_biography", "description", string_not_empty, features_groups=[UsersFeaturesFile, class_A])
has_at_least_30_followers = Feature("has_at_least_30_followers", "followers_count", lambda x: x >= 30, features_groups=[UsersFeaturesFile, class_A])
has_been_listed = Feature("has_been_listed", "listed_count", lambda x: x > 0, features_groups=[UsersFeaturesFile, class_A])
has_at_least_50_tweets = Feature("has_at_least_50_tweets", "statuses_count", lambda x: x >= 50, features_groups=[UsersFeaturesFile, class_A])
has_enabled_geoloc = Feature("has_enabled_geoloc", "geo_enabled", lambda x: x == 1, features_groups=[UsersFeaturesFile, class_B])
has_url = Feature("has_url", "url", string_not_empty, features_groups=[UsersFeaturesFile, class_A])
has_2followers_friends = Feature("has_2followers_friends", ["followers_count", "friends_count"], lambda fol, fri: 2*fol >= fri, features_groups=[UsersFeaturesFile, class_A])

explicit_biography = Feature("explicit_biography", "description", lambda x: "bot" in x, features_groups=[UsersFeaturesFile, class_A])
has_ratio_followers_friends_100 = Feature("has_ratio_followers_friends_100", ["followers_count", "friends_count"], lambda fol, fri: abs(fol - 100 * fri) <= 5 * fri, features_groups=[UsersFeaturesFile, class_A])
# duplicate_pictures

has_ratio_friends_followers_50 = Feature("has_ratio_friends_followers_50", ["followers_count", "friends_count"], lambda fol, fri: fri >= 50*fol, features_groups=[UsersFeaturesFile, class_A])
no_bio_no_location_friends_100 = Feature("no_bio_no_location_friends_100", ["description", "location", "friends_count"], lambda bio, loc, fri: string_empty(bio) and string_empty(loc) and fri >= 100, features_groups=[UsersFeaturesFile, class_A])
has_0_tweet = Feature("has_0_tweet", "statuses_count", lambda x: x == 0, features_groups=[UsersFeaturesFile, class_A])
# default_image_after_2_months

number_of_friends = Feature("number_of_friends", "friends_count", identity, features_groups=[UsersFeaturesFile, class_A])
number_of_tweets = Feature("number_of_tweets", "statuses_count", identity, features_groups=[UsersFeaturesFile, class_A])
ratio_friends_followers_square = Feature("ratio_friends_followers_square", ["followers_count", "friends_count"], lambda fol, fri: 0 if fol == 0 else fri / (fol * fol), features_groups=[UsersFeaturesFile, class_A])


# age
# following rate