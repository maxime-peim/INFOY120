import os
import re
import string
from collections import namedtuple
from collections.abc import Hashable, MutableSet
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd

ColumnToArg = namedtuple("ColumnToArg", ["column_name", "argument_name"])


class NamedSet(Hashable, MutableSet):
    __hash__ = MutableSet._hash

    def __init__(self, name="Generic group", iterable=()):
        self._name = name
        self.data = set(iterable)

    @property
    def name(self):
        return self._name

    def __contains__(self, value):
        return value in self.data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __or__(self, other):
        if not isinstance(other, (set, NamedSet, type(self))):
            raise NotImplementedError

        return type(self)(
            name=f"{self.name} U {other.name}", iterable=self.data.union(other)
        )

    def add(self, element):
        self.data.add(element)

    def discard(self, element):
        return self.data.discard(element)

    def __str__(self):
        return f"{self.name} = {{{', '.join(str(d) for d in self.data)}}}"


class FeaturesGroup(NamedSet):
    @property
    def names(self):
        return [feature.name for feature in self]


class Feature:
    registered = {}

    def __init__(self, name, evaluate, *, complex=False, **columns):
        self.name = name
        self._to_extract = self._build_extraction_dict(columns)
        self._evaluate = self._evaluate_proxy(evaluate, complex)
        self.registered[self.name] = self

    @property
    def evaluate(self):
        return self._evaluate

    def needed_columns(self, datafile_prefix):
        columns_to_args = self._to_extract.get(datafile_prefix, [])
        return [column_to_arg.column_name for column_to_arg in columns_to_args]

    def _evaluate_proxy(self, evaluate, complex):

        arg_column_mapping = {
            column_to_arg.argument_name: f"{class_prefix}_{column_to_arg.column_name}"
            for class_prefix, columns_to_args in self._to_extract.items()
            for column_to_arg in columns_to_args
        }

        def evaluate_row(row):
            kwargs = {arg: row[column] for arg, column in arg_column_mapping.items()}
            return evaluate(**kwargs)

        def evaluate_wrapper(dataframe):
            if complex:
                result = evaluate(dataframe)
            else:
                result = dataframe.apply(evaluate_row, axis=1)

            result.rename(self.name, inplace=True)
            return result

        return evaluate_wrapper

    @staticmethod
    def _build_extraction_dict(columns):
        result = {}
        for arg_name, column in columns.items():
            class_prefix, column_name = column.split("/")
            if class_prefix not in result:
                result[class_prefix] = []
            result[class_prefix].append(ColumnToArg(column_name, arg_name))

        return result

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError

        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Feature('{self.name}')"


def string_empty(string_arg):
    return len(str(string_arg).strip()) == 0


def string_not_empty(string_arg):
    return len(str(string_arg).strip()) > 0


def identity(x):
    return x


has_name = Feature("has_name", string_not_empty, string_arg="users/name")
has_image = Feature(
    "has_image", lambda default: default != 1, default="users/default_profile_image"
)
has_address = Feature("has_address", string_not_empty, string_arg="users/location")
has_biography = Feature(
    "has_biography", string_not_empty, string_arg="users/description"
)
has_at_least_30_followers = Feature(
    "has_at_least_30_followers", lambda fol: fol >= 30, fol="users/followers_count"
)
has_been_listed = Feature(
    "has_been_listed", lambda lis: lis > 0, lis="users/listed_count"
)
has_at_least_50_tweets = Feature(
    "has_at_least_50_tweets", lambda sta: sta >= 50, sta="users/statuses_count"
)
has_url = Feature("has_url", string_not_empty, string_arg="users/url")
has_2followers_friends = Feature(
    "has_2followers_friends",
    lambda fol, fri: 2 * fol >= fri,
    fol="users/followers_count",
    fri="users/friends_count",
)

explicit_biography = Feature(
    "explicit_biography", lambda bio: "bot" in bio, bio="users/description"
)
has_ratio_followers_friends_100 = Feature(
    "has_ratio_followers_friends_100",
    lambda fol, fri: abs(fol - 100 * fri) <= 5 * fri,
    fol="users/followers_count",
    fri="users/friends_count",
)


def duplicate_pictures_func(dataframe):
    images_df = dataframe["users_profile_image_url"].map(lambda x: os.path.basename(x))
    return images_df.groupby(images_df).transform(len).map(lambda x: x > 1)


duplicate_pictures = Feature(
    "duplicate_pictures",
    duplicate_pictures_func,
    complex=True,
    prof_img="users/profile_image_url",
)

has_ratio_friends_followers_50 = Feature(
    "has_ratio_friends_followers_50",
    lambda fol, fri: fri >= 50 * fol,
    fol="users/followers_count",
    fri="users/friends_count",
)
no_bio_no_location_friends_100 = Feature(
    "no_bio_no_location_friends_100",
    lambda bio, loc, fri: string_empty(bio) and string_empty(loc) and fri >= 100,
    bio="users/description",
    loc="users/location",
    fri="users/friends_count",
)
has_0_tweet = Feature("has_0_tweet", lambda sta: sta == 0, sta="users/statuses_count")


def age_func(crea):
    paper_date = datetime(2015, 2, 14)
    creation_date = datetime.strptime(crea, "%a %b %d %H:%M:%S +0000 %Y")

    return (creation_date - paper_date).days


def default_image_after_2_months_func(crea, default):
    if not default:
        return False

    # roughly 2 months
    return age_func(crea) >= 60


default_image_after_2_months = Feature(
    "default_image_after_2_months",
    default_image_after_2_months_func,
    crea="users/created_at",
    default="users/default_profile_image",
)

number_of_friends = Feature("number_of_friends", identity, x="users/friends_count")
number_of_tweets = Feature("number_of_tweets", identity, x="users/statuses_count")
ratio_friends_followers_square = Feature(
    "ratio_friends_followers_square",
    lambda fol, fri: 0 if fol == 0 else fri / (fol * fol),
    fol="users/followers_count",
    fri="users/friends_count",
)


age = Feature("age", age_func, crea="users/created_at")
following_rate = Feature(
    "following_rate",
    lambda crea, fri: fri / age_func(crea),
    crea="users/created_at",
    fri="users/friends_count",
)

class_A = FeaturesGroup(
    "Class A",
    [
        has_name,
        has_image,
        has_address,
        has_biography,
        has_at_least_30_followers,
        has_been_listed,
        has_at_least_50_tweets,
        has_url,
        has_2followers_friends,
        explicit_biography,
        has_ratio_followers_friends_100,
        duplicate_pictures,
        has_ratio_friends_followers_50,
        no_bio_no_location_friends_100,
        has_0_tweet,
        default_image_after_2_months,
        number_of_friends,
        number_of_tweets,
        ratio_friends_followers_square,
        age,
        following_rate,
    ],
)

has_enabled_geoloc = Feature(
    "has_enabled_geoloc", lambda geo: geo == 1, geo="users/geo_enabled"
)
is_favourite = Feature(
    "is_favourite", lambda fav: fav > 0, fav="users/favourites_count"
)

PUNCTUATION_PATTERN = re.compile("[" + re.escape(string.punctuation) + "]")
uses_punctuation = Feature(
    "uses_punctuation",
    lambda tweets: False
    if tweets is np.nan
    else any(PUNCTUATION_PATTERN.search(tweet) is not None for tweet in tweets),
    tweets="tweets/text",
)
uses_hashtag = Feature(
    "uses_hashtag",
    lambda tweets: False if tweets is np.nan else any("#" in tweet for tweet in tweets),
    tweets="tweets/text",
)


def uses_platform_func(source, platform):
    return platform.lower() in source.lower()


def is_platform_max(sources, platform_func):
    return (
        False if sources is np.nan else platform_func(max(sources, key=sources.count))
    )


uses_iphone_func = partial(uses_platform_func, platform="iphone")
uses_android_func = partial(uses_platform_func, platform="android")
uses_instagram_func = partial(uses_platform_func, platform="instagram")
uses_foursquare_func = partial(uses_platform_func, platform="foursquare")
uses_twitter_func = partial(uses_platform_func, platform="web")


def uses_other_func(source):
    return not (
        uses_iphone_func(source)
        or uses_android_func(source)
        or uses_instagram_func(source)
        or uses_foursquare_func(source)
        or uses_twitter_func(source)
    )


def uses_api_func(source):
    return not uses_twitter_func(source)


uses_iphone = Feature(
    "uses_iphone",
    partial(is_platform_max, platform_func=uses_iphone_func),
    sources="tweets/source",
)
uses_android = Feature(
    "uses_android",
    partial(is_platform_max, platform_func=uses_android_func),
    sources="tweets/source",
)
uses_instagram = Feature(
    "uses_instagram",
    partial(is_platform_max, platform_func=uses_instagram_func),
    sources="tweets/source",
)
uses_foursquare = Feature(
    "uses_foursquare",
    partial(is_platform_max, platform_func=uses_foursquare_func),
    sources="tweets/source",
)
uses_twitter = Feature(
    "uses_twitter",
    partial(is_platform_max, platform_func=uses_twitter_func),
    sources="tweets/source",
)
uses_other = Feature(
    "uses_other",
    partial(is_platform_max, platform_func=uses_other_func),
    sources="tweets/source",
)
user_id_in_tweets = Feature(
    "user_id_in_tweets",
    lambda tweets: False if tweets is np.nan else any("@" in tweet for tweet in tweets),
    tweets="tweets/text",
)
urls_in_tweets = Feature(
    "user_id_tweets",
    lambda tweets: False
    if tweets is np.nan
    else any("http" in tweet for tweet in tweets),
    tweets="tweets/text",
)
at_least_1_retweets = Feature(
    "at_least_1_retweets",
    lambda tweets: False
    if tweets is np.nan
    else any(tweet.startswith("RT @") for tweet in tweets),
    tweets="tweets/text",
)
same_tweets = Feature(
    "same_tweets",
    lambda tweets: False if tweets is np.nan else len(set(tweets)) != len(tweets),
    tweets="tweets/text",
)
same_tweets_3 = Feature(
    "same_tweets_3",
    lambda tweets: False
    if tweets is np.nan
    else tweets.count(max(tweets, key=tweets.count)) >= 3,
    tweets="tweets/text",
)


def spam_tweets_func(dataframe):
    tweets_df = dataframe.explode("tweets_text")["tweets_text"]
    return (
        tweets_df.transform(lambda x: 0 if x is np.nan else len(x))
        .groupby("user_id")
        .agg(max)
    )


spam_tweets = Feature(
    "spam_tweets", spam_tweets_func, complex=True, tweets="tweets/text"
)
retweet_90 = Feature(
    "retweet_90",
    lambda tweets: 0
    if tweets is np.nan
    else sum(tweet.startswith("RT @") for tweet in tweets) >= 0.9 * len(tweets),
    tweets="tweets/text",
)
urls_90 = Feature(
    "urls_90",
    lambda tweets: False
    if tweets is np.nan
    else sum("http" in tweet for tweet in tweets) >= 0.9 * len(tweets),
    tweets="tweets/text",
)
urls_ratio = Feature(
    "urls_ratio",
    lambda tweets: 0
    if tweets is np.nan
    else sum("http" in tweet for tweet in tweets) / len(tweets),
    tweets="tweets/text",
)


def tweets_similarity_func(tweets, created):
    if tweets is np.nan or created is np.nan:
        return False

    last_15 = [
        t.lower()
        for t, _ in sorted(
            zip(tweets, created),
            key=lambda x: datetime.strptime(x[1], "%a %b %d %H:%M:%S +0000 %Y"),
        )
    ][-15:]
    consecutives = []
    for tweet in last_15:
        four_consecutive = []
        for word in tweet.split():
            four_consecutive.append(word)
            if len(four_consecutive) > 4:
                four_consecutive.pop(0)

            word4 = "".join(four_consecutive)
            if word4 in consecutives:
                return True

            if len(four_consecutive) == 4:
                consecutives.append(word4)
    return False


tweets_similarity = Feature(
    "tweets_similarity",
    tweets_similarity_func,
    tweets="tweets/text",
    created="tweets/created_at",
)
api_ratio = Feature(
    "api_ratio",
    lambda sources: 0
    if sources is np.nan
    else sum(uses_api_func(source) for source in sources) / len(sources),
    sources="tweets/source",
)
uses_api = Feature(
    "uses_api",
    partial(is_platform_max, platform_func=uses_api_func),
    sources="tweets/source",
)
api_urls_ratio = Feature(
    "api_urls_ratio",
    lambda tweets, sources: 0
    if sources is np.nan or tweets is np.nan
    else sum(
        "http" in tweet and uses_api_func(source)
        for tweet, source in zip(tweets, sources)
    )
    / len(tweets),
    tweets="tweets/text",
    sources="tweets/source",
)
api_tweets_similarity = Feature(
    "api_tweets_similarity",
    lambda tweets, created, sources: is_platform_max(sources, uses_api_func)
    and tweets_similarity_func(tweets, created),
    tweets="tweets/text",
    created="tweets/created_at",
    sources="tweets/source",
)

class_B = FeaturesGroup(
    "Class B",
    [
        has_enabled_geoloc,
        is_favourite,
        uses_punctuation,
        uses_hashtag,
        uses_iphone,
        uses_android,
        uses_instagram,
        uses_foursquare,
        uses_twitter,
        uses_other,
        user_id_in_tweets,
        urls_in_tweets,
        at_least_1_retweets,
        same_tweets,
        same_tweets_3,
        spam_tweets,
        retweet_90,
        urls_90,
        urls_ratio,
        tweets_similarity,
        api_ratio,
        uses_api,
        api_urls_ratio,
        api_tweets_similarity,
    ],
)


def bidirectional_link_ratio_func(dataframe):
    def bidirectional_ratio(friends, followers):
        # if a user has no friend or follower
        if friends is np.nan or followers is np.nan:
            return 0
        # else get the intersection of his friends and followers
        # to have the number of bidirectional links
        return len((set(friends)).intersection(set(followers))) / len(friends)

    return dataframe.apply(
        lambda row: bidirectional_ratio(
            row.friends_friends_id, row.followers_followers_id
        ),
        axis=1,
    )


bidirectional_link_ratio = Feature(
    "bidirectional_link_ratio",
    bidirectional_link_ratio_func,
    complex=True,
    friends="friends/friends_id",
    followers="followers/followers_id",
)


def average_neighbors_followers_func(dataframe):
    def mean_followers_of_friends(friends, followers_count):
        if friends is np.nan or followers_count is np.nan:
            return 0
        friends_index = pd.Index(set(friends))
        index = friends_index.intersection(followers_count.index)
        return followers_count.loc[index].mean()

    mean_followers = partial(
        mean_followers_of_friends, followers_count=dataframe["users_followers_count"]
    )
    return dataframe["friends_friends_id"].apply(
        lambda friends: mean_followers(friends)
    )


average_neighbors_followers = Feature(
    "average_neighbor_followers",
    average_neighbors_followers_func,
    complex=True,
    friends="friends/friends_id",
    followers_count="users/followers_count",
)


def average_neighbors_tweets_func(dataframe):
    def mean_tweets_of_followers(followers, tweets_count):
        if followers is np.nan or tweets_count is np.nan:
            return 0
        followers_index = pd.Index(set(followers))
        index = followers_index.intersection(tweets_count.index)
        return tweets_count.loc[index].mean()

    mean_tweets = partial(
        mean_tweets_of_followers, tweets_count=dataframe["users_statuses_count"]
    )
    return dataframe["followers_followers_id"].apply(
        lambda followers: mean_tweets(followers)
    )


average_neighbors_tweets = Feature(
    "average_neighbors_tweets",
    average_neighbors_tweets_func,
    complex=True,
    followers="followers/followers_id",
    tweets_count="users/statuses_count",
)


def following_to_median_followers_func(dataframe):
    def friends_to_median_followers_of_friends(friends, followers_count):
        if friends is np.nan or followers_count is np.nan:
            return 0
        friends_index = pd.Index(set(friends))
        index = friends_index.intersection(followers_count.index)
        median = followers_count.loc[index].median()
        return 0 if median == 0 else len(friends_index) / median

    median_followers_ratio = partial(
        friends_to_median_followers_of_friends,
        followers_count=dataframe["users_followers_count"],
    )
    return dataframe["friends_friends_id"].apply(
        lambda friends: median_followers_ratio(friends)
    )


following_to_median_followers = Feature(
    "following_to_median_followers",
    following_to_median_followers_func,
    complex=True,
    friends="friends/friends_id",
    followers_count="users/followers_count",
)

class_C = FeaturesGroup(
    "Class C",
    [
        bidirectional_link_ratio,
        average_neighbors_followers,
        average_neighbors_tweets,
        following_to_median_followers,
    ],
)

all_features = FeaturesGroup("All", class_A | class_B | class_C)
set_Y = FeaturesGroup(
    "Set Yang and al",
    [
        age,
        bidirectional_link_ratio,
        average_neighbors_followers,
        average_neighbors_tweets,
        following_to_median_followers,
        api_ratio,
        api_urls_ratio,
        api_tweets_similarity,
        following_rate,
    ],
)

set_S = FeaturesGroup(
    "Set Stringhini and al",
    [
        number_of_friends,
        number_of_tweets,
        tweets_similarity,
        urls_ratio,
        ratio_friends_followers_square,
    ],
)
