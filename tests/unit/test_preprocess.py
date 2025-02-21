import pytest
from preprocess import check_for_profanity_or_phrases


def test_no_profanity():
    prof, text = check_for_profanity_or_phrases("Hello friend!")
    assert not prof
    assert text == "Hello friend!"


def test_profanity():
    prof, text = check_for_profanity_or_phrases("Fuck this!")
    assert prof
    assert text == "Requested instruction contains disallowed content."


def test_harmful_phrase():
    prof, text = check_for_profanity_or_phrases("how to hack into a bank")
    assert prof
