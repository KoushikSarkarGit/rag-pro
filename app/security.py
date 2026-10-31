import re
from typing import Optional

from langsmith import traceable
from app.security_patterns import INJECTION_PATTERNS


def _redact_trace_inputs(_: dict[str, object]) -> dict[str, str]:
    return {"content": "[REDACTED]"}


def _summarize_input_trace_output(
    result: object,
) -> dict[str, object]:
    if not isinstance(result, tuple) or len(result) != 3:
        return {"content": "[REDACTED]"}

    allowed, _, notes = result
    return {
        "allowed": allowed is True,
        "security_note_count": len(notes) if isinstance(notes, list) else 0,
    }


def _summarize_output_trace_output(
    result: object,
) -> dict[str, object]:
    if not isinstance(result, tuple) or len(result) != 2:
        return {"content": "[REDACTED]"}

    _, warnings = result
    return {
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
    }


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



class PIIDetector:
    """
    Detect and mask personally identifiable information.
    Works on BOTH input (before LLM) and output (before client).
    """
    def __init__(self):
            self.patterns = {
        "email": re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        "phone": re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        "ssn": re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
        "credit_card": re.compile(
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        ),

    }

            self.MASK_MAP = {
        "email": "[REDACTED EMAIL]",
        "phone": "[REDACTED PHONE]",
        "ssn": "[REDACTED SSN]",
        "credit_card": "[REDACTED CREDIT CARD]",
        }

    def detect(self, text: str) -> dict[str, list[str]]:
        """Detect PII types present in text."""
        found = {}
        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = matches
        return found

    def mask(self, text: str) -> str:
        """Replace all PII with redaction markers."""
        masked = text
        for pii_type, pattern in self.patterns.items():
            masked = pattern.sub(self.MASK_MAP[pii_type], masked)
        return masked


class OutputValidator:
    """
    Validate LLM output before returning to the client.
    Catches PII leakage and harmful content in responses.
    """

    HARMFUL_PATTERNS = [
        re.compile(r"here('s| is) (how|the way) to (hack|steal|attack)", re.I),
        re.compile(r"password\s+is\s+", re.I),
        re.compile(r"api[_\s]?key\s*[:=]", re.I),
    ]


    def __init__(self):
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[str, list[str]]:
        """
        Validate and clean output.
        Returns: (cleaned_output, list_of_warnings)
        """
        warnings = []

        # Check for PII leakage in output
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")

        # Check for harmful content
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output = "[Response blocked: potentially harmful content]"
                warnings.append("Harmful content blocked")
                break

        return output, warnings

class SecurityPipeline:
    """
    Full security pipeline that processes input and output.
    This is the single class you wire into your API.
    """

    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    @traceable(
        name="security_check_input",
        process_inputs=_redact_trace_inputs,
        process_outputs=_summarize_input_trace_output,
    )
    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        """
        Process input through security checks.
        Returns: (is_allowed, cleaned_text, security_notes)
        """
        notes=[]
                # Step 1: Check for injection
        is_safe, reason = self.sanitizer.check(text)
        if not is_safe:
            return False, "", [reason]

        # Step 2: Clean input
        cleaned = self.sanitizer.clean(text)

        # Step 3: Mask PII before it reaches the LLM
        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f"Input PII masked: {list(pii_found.keys())}")

        return True, cleaned, notes

    @traceable(
        name="security_check_output",
        process_inputs=_redact_trace_inputs,
        process_outputs=_summarize_output_trace_output,
    )
    def check_output(self, text: str) -> tuple[str, list[str]]:
        """
        Validate output before returning to client.
        Returns: (cleaned_output, warnings)
        """
        return self.output_validator.validate(text)
