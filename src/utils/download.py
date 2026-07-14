from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def download_file(url: str, dest: Path, retries: int = 10) -> None:
    for attempt in range(1, retries + 1):
        logger.info(f"  [{attempt}/{retries}] {dest.name}")
        result = subprocess.run(
            [
                "wget",
                "--continue",
                "--retry-connrefused",
                "--timeout=30",
                "--waitretry=5",
                "--tries=5",
                "-q",
                "-O",
                str(dest),
                url,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info(f"  ✓ {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return
        logger.warning(f"  ✗ {dest.name} failed (attempt {attempt}): {result.stderr.strip()}")
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")
