from better_profanity import profanity


BANNED_PHRASES = [
    "how to kill",
    "how to hack",
    "bomb-making",
]


def check_for_banned_phrases(text: str) -> bool:
    lower_text = text.lower()
    for phrase in BANNED_PHRASES:
        if phrase in lower_text:
            return True
    return False


def check_for_profanity_or_phrases(input_text: str) -> tuple[bool, str]:
    if profanity.contains_profanity(input_text) or check_for_banned_phrases(input_text):
        return True, "Requested instruction contains disallowed content."
    else:
        return False, input_text
