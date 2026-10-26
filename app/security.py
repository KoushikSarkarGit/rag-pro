import re
from typing import Optional

from app.security_patterns import INJECTION_PATTERNS


class InputSanitizer():

    def __init__(self):
            self.patterns = [re.compile(pattern, re.IGNORECASE) for pattern in INJECTION_PATTERNS]

    def check(self, text: str) -> tuple[bool, Optional[str]]:
            """
            Check if input is safe.
            Returns: (is_safe, rejection_reason)
            """
            for pattern in self.patterns:
                if pattern.search(text):
                    return False, "Blocked: potential prompt injection detected"
            return True, None


    def clean(self, text: str) -> str:
        """Remove potentially dangerous delimiters from input."""
        text = re.sub(r'[-]{3,}', '', text)
        text = re.sub(r'[=]{3,}', '', text)
        text = text.replace('{{', '{ {').replace('}}', '} }')
        return text.strip()
