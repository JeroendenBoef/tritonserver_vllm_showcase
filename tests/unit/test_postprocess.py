import pytest
from model_gaurdrails import censor_profanity


def test_no_profanity():
    original_text = "Hello friend!"
    censored_text = censor_profanity(original_text)
    assert original_text == censored_text


def test_profanity():
    original_text = "Fuck this!"
    censored_text = censor_profanity("Fuck this!")
    assert original_text != censored_text
    assert censored_text == "---- this!"
