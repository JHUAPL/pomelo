import pandas as pd

from harddisc.feature_extraction.extractors.date_encoder import DateEncoder


def test_date_encoding():
    dates = ["2/5/2003 12:00", "2/7/2008 10:45", "6/4/2007 13:00"]

    date_series = pd.Series(dates)

    date_encoder = DateEncoder()

    encoded_dates = date_encoder.encode(date_series)

    print(encoded_dates)

    assert encoded_dates.shape == (3,)  # noqa: S101
