"""
ResearchFlow — PII Masking Middleware

Scans user inputs and agent outputs for personally identifiable
information and redacts it before processing or returning.
"""

import re


# Patterns to detect — extend as needed
PII_PATTERNS: dict[str, str] = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}


def mask_pii(text: str) -> str:
    """
    Replace detected PII patterns with redaction placeholders.

    TODO:
    - Iterate over PII_PATTERNS and apply re.sub.
    - Return the sanitized text.
    - Consider logging redaction counts to the scratchpad.
    """

    # use the PII patterns given to us to replace PII
    text = re.sub(PII_PATTERNS["email"], "[REDACTED_EMAIL]", text)
    text = re.sub(PII_PATTERNS["phone"], "[REDACTED_PHONE]", text)
    text = re.sub(PII_PATTERNS["ssn"], "[REDACTED_SSN]", text)

    return text