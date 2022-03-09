from collections import namedtuple

ColumnToArg = namedtuple("ColumnToArg", ["column_name", "argument_name"])


class FeaturesGroup:
    def __init__(self, *features, name="Generic group"):
        self.name = name

        self._features = set(features)
        for feature in self._features:
            self._features.add(feature)

    @property
    def features(self):
        return self._features

    def add_feature(self, feature):
        self._features.add(feature)

    def __iter__(self):
        return self._features.__iter__()

    def __next__(self):
        return self._features.__next__()


class Feature:
    def __init__(self, name, evaluate, *, complex=False, **columns):
        self.name = name
        self._to_extract = self._build_extraction_dict(columns)
        self._evaluate = self._evaluate_proxy(evaluate, complex)

        self._groups = []

    @property
    def features_groups(self):
        return self._groups

    @property
    def evaluate(self):
        return self._evaluate

    def needed_columns(self, datafile_prefix):
        columns_to_args = self._to_extract.get(datafile_prefix, [])
        return [column_to_arg.column_name for column_to_arg in columns_to_args]

    def add_to_group(self, group):
        self._groups.append(group)

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

    def __str__(self):
        return f"Feature {self.name} in groups {'/'.join(group.name for group in self._groups)}"


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
    "explicit_biography", lambda desc: "bot" in desc, desc="users/description"
)
has_ratio_followers_friends_100 = Feature(
    "has_ratio_followers_friends_100",
    lambda fol, fri: abs(fol - 100 * fri) <= 5 * fri,
    fol="users/followers_count",
    fri="users/friends_count",
)
# duplicate_pictures

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
has_0_tweet = Feature("has_0_tweet", lambda x: x == 0, x="users/statuses_count")
# default_image_after_2_months

number_of_friends = Feature("number_of_friends", identity, x="users/friends_count")
number_of_tweets = Feature("number_of_tweets", identity, x="users/statuses_count")
ratio_friends_followers_square = Feature(
    "ratio_friends_followers_square",
    lambda fol, fri: 0 if fol == 0 else fri / (fol * fol),
    fol="users/followers_count",
    fri="users/friends_count",
)


# age
# following rate

class_A = FeaturesGroup(
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
    has_ratio_friends_followers_50,
    no_bio_no_location_friends_100,
    has_0_tweet,
    number_of_friends,
    number_of_tweets,
    ratio_friends_followers_square,
    name="Class A",
)
class_B = FeaturesGroup(name="Class B")
class_C = FeaturesGroup(name="Class C")
set_CC = FeaturesGroup(name="Set CC")
