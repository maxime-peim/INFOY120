import os
from collections import namedtuple
from collections.abc import Hashable, Set
from datetime import datetime

ColumnToArg = namedtuple("ColumnToArg", ["column_name", "argument_name"])


class NamedSet(Hashable, Set):
    __hash__ = Set._hash

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


FeaturesGroup = NamedSet


class Feature:
    def __init__(self, name, evaluate, *, complex=False, **columns):
        self.name = name
        self._to_extract = self._build_extraction_dict(columns)
        self._evaluate = self._evaluate_proxy(evaluate, complex)

    @property
    def evaluate(self):
        return self._evaluate

    def needed_columns(self, datafile_prefix):
        columns_to_args = self._to_extract.get(datafile_prefix, [])
        return [column_to_arg.column_name for column_to_arg in columns_to_args]

    def _evaluate_proxy(self, evaluate, complex):
        if complex:
            return evaluate

        arg_column_mapping = {
            column_to_arg.argument_name: f"{class_prefix}_{column_to_arg.column_name}"
            for class_prefix, columns_to_args in self._to_extract.items()
            for column_to_arg in columns_to_args
        }

        def evaluate_wrapper(dataframe):
            def evaluate_row(row):
                kwargs = {
                    arg: row[column] for arg, column in arg_column_mapping.items()
                }
                return evaluate(**kwargs)

            return dataframe.apply(evaluate_row, axis=1)

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
has_enabled_geoloc = Feature(
    "has_enabled_geoloc", lambda geo: geo == 1, geo="users/geo_enabled"
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
        has_enabled_geoloc,
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
class_B = FeaturesGroup("Class B")
class_C = FeaturesGroup("Class C")
set_CC = FeaturesGroup("Set CC")
