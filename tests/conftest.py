"""Shared fixtures for catch_joe tests."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest


# ── Minimal in-memory session data ───────────────────────────────────────────

RAW_SESSIONS = [
    {
        "user_id": 0,
        "browser": "Chrome",
        "os": "Windows 10",
        "locale": "en-US",
        "gender": "m",
        "location": "USA/New York",
        "date": "2019-01-01",
        "time": "10:00:00",
        "sites": [
            {"site": "github.com", "length": 120},
            {"site": "stackoverflow.com", "length": 80},
            {"site": "github.com", "length": 60},
        ],
    },
    {
        "user_id": 0,
        "browser": "Chrome",
        "os": "Windows 10",
        "locale": "en-US",
        "gender": "m",
        "location": "USA/New York",
        "date": "2019-01-02",
        "time": "11:00:00",
        "sites": [
            {"site": "github.com", "length": 200},
            {"site": "python.org", "length": 90},
        ],
    },
    {
        "user_id": 1,
        "browser": "Firefox",
        "os": "Linux",
        "locale": "de-DE",
        "gender": "f",
        "location": "Germany/Berlin",
        "date": "2019-01-03",
        "time": "08:30:00",
        "sites": [
            {"site": "google.com", "length": 30},
            {"site": "youtube.com", "length": 300},
        ],
    },
    {
        "user_id": 2,
        "browser": "Safari",
        "os": "macOS",
        "locale": "fr-FR",
        "gender": "f",
        "location": "France/Paris",
        "date": "2019-01-04",
        "time": "14:00:00",
        "sites": [
            {"site": "wikipedia.org", "length": 150},
            {"site": "github.com", "length": 50},
        ],
    },
    {
        "user_id": 1,
        "browser": "Firefox",
        "os": "Linux",
        "locale": "de-DE",
        "gender": "f",
        "location": "Germany/Berlin",
        "date": "2019-01-05",
        "time": "09:00:00",
        "sites": [
            {"site": "youtube.com", "length": 400},
            {"site": "google.com", "length": 20},
            {"site": "python.org", "length": 70},
        ],
    },
]


@pytest.fixture()
def raw_sessions() -> list[dict]:
    return RAW_SESSIONS


@pytest.fixture()
def dataset_json(tmp_path: Path) -> Path:
    """Write RAW_SESSIONS to a temp JSON file and return the path."""
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(RAW_SESSIONS))
    return p


@pytest.fixture()
def verify_json(tmp_path: Path) -> Path:
    """Write sessions without user_id to a temp JSON file."""
    sessions_no_id = [
        {k: v for k, v in s.items() if k != "user_id"} for s in RAW_SESSIONS
    ]
    p = tmp_path / "verify.json"
    p.write_text(json.dumps(sessions_no_id))
    return p


@pytest.fixture()
def sessions_df(dataset_json: Path) -> pd.DataFrame:
    """Loaded DataFrame from dataset_json fixture."""
    from catch_joe.data import load_sessions
    return load_sessions(dataset_json)


@pytest.fixture()
def sessions_with_target(sessions_df: pd.DataFrame) -> pd.DataFrame:
    from catch_joe.data import create_target
    from catch_joe.features import extract_session_stats
    df = create_target(sessions_df, target_user_id=0)
    return extract_session_stats(df)
