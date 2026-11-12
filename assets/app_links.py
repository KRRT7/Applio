"""Public identity and navigation for the Gradio interface; no engine imports."""

import json
from pathlib import Path
from urllib.parse import urljoin

_ASSETS = Path(__file__).resolve().parent
ECOSYSTEM = json.loads((_ASSETS / "ecosystem.json").read_text(encoding="utf-8"))


def guide_url(guide):
    return urljoin(ECOSYSTEM["links"]["docs"], ECOSYSTEM["guides"][guide])
