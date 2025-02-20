from better_profanity import profanity


def censor_profanity(input_text: str) -> str:
    return profanity.censor(input_text, "-") if profanity.contains_profanity(input_text) else input_text
