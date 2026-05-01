"""
ResearchFlow — Input Guardrails Middleware

Detects and blocks prompt injection / stuffing attacks
in user inputs before they reach the agent pipeline.
"""
import re

# regular expressions for common injection patterns i found online
INJECTION_PATTERNS = [
    r"ignore (all|previous|above) instructions",
    r"system prompt",
    r"you are now",
    r"act as (a|an)",
    r"disregard (all|previous)",
    r"override",
    r"jailbreak",
    r"developer mode",
    r"<\s*script\s*>",
]
def detect_injection(user_input: str) -> bool:
    """
    Scan user input for common prompt injection patterns.

    TODO:
    - Check for system prompt override attempts.
    - Check for instruction stuffing patterns.
    - Return True if injection is detected, False otherwise.
    """
    text = user_input.lower()

    # pattern match
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text):
            return True

    # excessive instruction density heuristic
    if text.count("!") > 5:
        return True

    if len(re.findall(r"(system:|assistant:|user:)", text)) > 2:
        return True

    return False


def sanitize_input(user_input: str) -> str:
    """
    Clean user input by removing or escaping dangerous patterns.

    TODO:
    - Strip known injection markers.
    - Escape special formatting that could manipulate prompts.
    - Return the sanitized string.
    """
    text = user_input.lower()

    # If injection detected then remove matched patterns
    for pattern in INJECTION_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # remove any role prefixes
    text = re.sub(r"(system:|assistant:|user:)", "", text, flags=re.IGNORECASE)

    # Remove script stuff
    text = re.sub(
        r"<\s*script.*?>.*?<\s*/\s*script\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
