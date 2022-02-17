import pandas as pd
import scipy.stats

# Non Deterministic Test
def test_kolmogorov_smirnov(data, ks_alpha):

    sample1, sample2 = data

    numerical_columns = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous"
    ]
    
    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:

        # two-sided: The null hypothesis is that the two distributions are identical
        # the alternative is that they are not identical.
        ts, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime
        
# Determinstic Test
def test_column_presence_and_type(data):
    
    # Disregard the reference dataset
    _, df = data

    required_columns = {
        "age": pd.api.types.is_int64_dtype,
        "job": pd.api.types.is_object_dtype,
        "marital": pd.api.types.is_object_dtype,
        "education": pd.api.types.is_object_dtype,
        "default": pd.api.types.is_object_dtype,
        "balance": pd.api.types.is_int64_dtype,
        "housing": pd.api.types.is_object_dtype,
        "loan": pd.api.types.is_object_dtype,
        "contact": pd.api.types.is_object_dtype,
        "day": pd.api.types.is_int64_dtype,
        "month": pd.api.types.is_object_dtype,
        "duration": pd.api.types.is_int64_dtype,  
        "campaign": pd.api.types.is_int64_dtype,
        "pdays": pd.api.types.is_int64_dtype,
        "previous": pd.api.types.is_int64_dtype,
        "poutcome": pd.api.types.is_object_dtype,
        "y": pd.api.types.is_object_dtype
    }

    # Check column presence
    assert set(df.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(df[col_name]), f"Column {col_name} failed test {format_verification_funct}"

# Deterministic Test
def test_class_names(data):
    
    # Disregard the reference dataset
    _, df = data

    # Check that only the known classes are present
    known_classes = [
        "yes",
        "no"
    ]

    assert df["y"].isin(known_classes).all()

# Deterministic Test
def test_column_ranges(data):
    
    # Disregard the reference dataset
    _, df = data

    ranges = {
        "age": (18, 96),
        "balance": (-10000, 120000),
        "day": (1, 31),
        "duration": (0, 4918),
        "campaign": (1, 63),
        "pdays": (-1, 871),
        "previous": (0, 275)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert df[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={df[col_name].min()} and max={df[col_name].max()}"
        )