import pytest

import pandas as pd

from harddisc.generative.dataset import GenerativeDataset
from harddisc.generative.generative_model_offline import GenerativeModelOffline


@pytest.fixture
def finished_prompts():
    df = pd.read_csv("test/test_data.csv").dropna().iloc[:5]

    dataset = GenerativeDataset(
        df, "{GPT_SUMMARY}\n Question: Is this person dead?\n Answer:"
    )

    return dataset.create_prompts()


def test_creation_default():
    model = GenerativeModelOffline("distilbert/distilgpt2", {"batch_size": 1})

    assert model is not None  # noqa: S101


def test_creation_not_default():
    model = GenerativeModelOffline("distilbert/distilgpt2", {"batch_size": 1, "temperature": 0.3})

    assert model is not None  # noqa: S101


def test_format_response():
    model = GenerativeModelOffline("distilbert/distilgpt2", {"batch_size": 1})

    test_dict = "Hello world!"

    output = model.format_response(test_dict)

    assert output == "Hello world!"  # noqa: S101


def test_get_response_default():
    model = GenerativeModelOffline("distilbert/distilgpt2", {"batch_size": 1})

    test_string = "Please tell me a story: "

    output = model.get_response(test_string)

    assert output is not None  # noqa: S101


def test_generate_from_prompts_default(finished_prompts):
    model = GenerativeModelOffline("distilbert/distilgpt2", {"batch_size": 1})

    output = model.generate_from_prompts(finished_prompts)

    assert output is not None  # noqa: S101
    assert len(output) == 5  # noqa: S101


def test_get_response_not_default():
    model = GenerativeModelOffline("distilbert/distilgpt2", {"batch_size": 1, "temperature": 0.3})

    test_string = "Please tell me a story: "

    output = model.get_response(test_string)

    assert output is not None  # noqa: S101


def test_generate_from_prompts_not_default(finished_prompts):
    model = GenerativeModelOffline("distilbert/distilgpt2", {"batch_size": 1, "temperature": 0.3})

    output = model.generate_from_prompts(finished_prompts)

    assert output is not None  # noqa: S101
    assert len(output) == 5  # noqa: S101
