INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(all\s+)?previous",
    r"new\s+instructions\s*:",
    r"system\s*prompt",
    r"--- \s*end\s*(of)?\s*prompt",
    r"pretend\s+you\s+are",
    r"act\s+as\s+(if\s+)?you",
    r"bypass\s+(all\s+)?restrictions",
    r"reveal\s+(your|the)\s+(system|instructions|prompt)",
    r"you\s+are\s+now\s+(DAN|jailbroken)",
)
