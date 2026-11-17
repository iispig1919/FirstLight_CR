"""Render a review document as one self-contained HTML page.

The page embeds the review JSON and draws everything in the browser with no
network access and no other file, so it opens straight from disk on the
machine that ran the review.
"""

from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Mapping


TEMPLATE_PATH = Path(__file__).with_name("page_template.html")
JSON_SLOT = "__REVIEW_JSON__"
TITLE_SLOT = "__REVIEW_TITLE__"


def render_page(review: Mapping[str, Any]) -> str:
    """Embed one review document into the report page template."""

    # Escaping "<" keeps any card name or failure text from closing the script element.
    payload = json.dumps(review, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if template.count(JSON_SLOT) != 1:
        raise RuntimeError("review page template must contain exactly one data slot")
    return template.replace(TITLE_SLOT, escape(str(review["replay_tag"]))).replace(JSON_SLOT, payload)
