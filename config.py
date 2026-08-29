"""
config.py
=========
Isolated configuration layer for the BizAgent Supermart Operations
Console.

Nothing in this module contains business/sample data. Its only job is
to read two external flat files and hand back typed, structured
objects:

    .env             -> credentials, DB path, reporting window sizes
    sop_policy.txt   -> escalation thresholds, narrative policy text,
                         and the keyword -> DatabaseManager-method
                         mapping that drives the Assistant tab

No external dependency (e.g. python-dotenv) is required; both files
use a simple, explicit format parsed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


# --------------------------------------------------------------------------- 
# .env parsing
# --------------------------------------------------------------------------- 

def _parse_env_file(path: Path) -> dict:
    """Parse a simple KEY=VALUE .env file into a dict of strings."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing configuration file: {path}. "
            "This application requires a .env file with credentials and "
            "settings — see the .env template shipped alongside app.py."
        )
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


@dataclass(frozen=True)
class Credential:
    username: str
    password: str
    role: str


@dataclass(frozen=True)
class AppConfig:
    app_title: str
    db_path: Path
    sop_policy_file: Path
    sales_lookback_days: int
    top_products_lookback_days: int
    top_products_limit: int
    recent_transactions_limit: int
    credentials: dict = field(default_factory=dict)  # username -> Credential

    def authenticate(self, username: str, password: str) -> Optional[Credential]:
        """Validate a login attempt against credentials loaded from .env."""
        cred = self.credentials.get(username)
        if cred and cred.password == password:
            return cred
        return None


def load_app_config(env_path: Path = ENV_PATH) -> AppConfig:
    """Read .env and return a fully-typed AppConfig."""
    raw = _parse_env_file(env_path)

    def _get(key: str, default: Optional[str] = None) -> str:
        if key not in raw and default is None:
            raise KeyError(f"Required setting '{key}' missing from {env_path}")
        return raw.get(key, default)

    credentials = {
        raw["ADMIN_USERNAME"]: Credential(
            username=raw["ADMIN_USERNAME"],
            password=raw["ADMIN_PASSWORD"],
            role=raw.get("ADMIN_ROLE", "Operations Manager"),
        ),
        raw["ASSOCIATE_USERNAME"]: Credential(
            username=raw["ASSOCIATE_USERNAME"],
            password=raw["ASSOCIATE_PASSWORD"],
            role=raw.get("ASSOCIATE_ROLE", "Store Associate"),
        ),
    }

    return AppConfig(
        app_title=_get("APP_TITLE", "BizAgent Operations Console"),
        db_path=(BASE_DIR / _get("DB_PATH", "supermart_ops.db")),
        sop_policy_file=(BASE_DIR / _get("SOP_POLICY_FILE", "sop_policy.txt")),
        sales_lookback_days=int(_get("SALES_LOOKBACK_DAYS", "7")),
        top_products_lookback_days=int(_get("TOP_PRODUCTS_LOOKBACK_DAYS", "30")),
        top_products_limit=int(_get("TOP_PRODUCTS_LIMIT", "5")),
        recent_transactions_limit=int(_get("RECENT_TRANSACTIONS_LIMIT", "20")),
        credentials=credentials,
    )


# --------------------------------------------------------------------------- 
# sop_policy.txt parsing
# --------------------------------------------------------------------------- 

@dataclass(frozen=True)
class IntentRule:
    """One assistant intent: a set of trigger keywords mapped to a live DB call."""
    name: str
    keywords: list
    method: str


@dataclass(frozen=True)
class SOPPolicy:
    escalation_lead_time_days: int
    slow_moving_window_days: int
    narrative: str
    intents: list  # list[IntentRule]

    def match_intent(self, user_text: str) -> Optional[IntentRule]:
        """
        Score each intent by how many of its keywords appear in the
        user's text and return the best match, or None if nothing hits.
        This is the entire "NLU" layer for the Assistant tab -- it is
        driven purely by sop_policy.txt, not by any hardcoded logic here.
        """
        text = user_text.lower()
        best: Optional[IntentRule] = None
        best_score = 0
        for intent in self.intents:
            score = sum(1 for kw in intent.keywords if kw in text)
            if score > best_score:
                best_score = score
                best = intent
        return best


_INTENT_LINE_RE = re.compile(
    r"INTENT:\s*(?P<name>[^|]+)\|\s*KEYWORDS:\s*(?P<keywords>[^|]+)\|\s*METHOD:\s*(?P<method>.+)"
)


def load_sop_policy(policy_path: Path) -> SOPPolicy:
    """Parse sop_policy.txt into a structured SOPPolicy object."""
    if not policy_path.exists():
        raise FileNotFoundError(
            f"Missing SOP policy file: {policy_path}. "
            "This application requires an external sop_policy.txt defining "
            "escalation thresholds and assistant intents."
        )

    text = policy_path.read_text(encoding="utf-8")

    section_pattern = re.compile(r"^\[(\w+)\]\s*$", re.MULTILINE)
    sections: dict[str, str] = {}
    matches = list(section_pattern.finditer(text))
    for i, match in enumerate(matches):
        name = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[start:end].strip()

    # --- [POLICY] key = value pairs ---
    policy_values: dict[str, str] = {}
    for line in sections.get("POLICY", "").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        policy_values[key.strip()] = value.strip()

    # --- [NARRATIVE] free text, shown verbatim in the UI ---
    narrative = sections.get("NARRATIVE", "").strip()

    # --- [INTENTS] machine-readable rule lines ---
    intents: list[IntentRule] = []
    for line in sections.get("INTENTS", "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _INTENT_LINE_RE.match(line)
        if not match:
            continue
        keywords = [kw.strip().lower() for kw in match.group("keywords").split(",") if kw.strip()]
        intents.append(
            IntentRule(
                name=match.group("name").strip(),
                keywords=keywords,
                method=match.group("method").strip(),
            )
        )

    return SOPPolicy(
        escalation_lead_time_days=int(policy_values.get("ESCALATION_LEAD_TIME_DAYS", "5")),
        slow_moving_window_days=int(policy_values.get("SLOW_MOVING_WINDOW_DAYS", "30")),
        narrative=narrative,
        intents=intents,
    )
