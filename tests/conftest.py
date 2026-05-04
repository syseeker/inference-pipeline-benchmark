"""Test fixtures.

`fake_image_bytes` keeps tests offline — we never want CI to hit a NIM
endpoint. Reasoner tests that need a real backend live behind the `nim`
or `gpu` markers and are skipped by default.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image


@pytest.fixture
def fake_image_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color=(127, 127, 127))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
