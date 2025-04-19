# tasks/extract_formants.py

import requests
import subprocess
import tempfile

from utils.vowel_files import vowel_files
from pathlib import Path
from urllib.parse import urlparse

OUTPUT_DIR = "vowel_encodings/formants"

for vowel_label, url in vowel_files:

    headers = {
        "User-Agent": "CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)"
    }
    response = requests.get(url, headers=headers)
    suffix = urlparse(url).path.split(".")[-1]

    response.raise_for_status()  # Raise error if download failed

    with tempfile.NamedTemporaryFile(suffix=f".{suffix}", delete=False) as tmp:
        tmp.write(response.content)
        temp_path = tmp.name

    print(f"Saved to: {temp_path}")

    result = subprocess.run(
        [
            "praat",
            "--run",
            "tasks/extract_formants.praat",
            temp_path,
            str(Path(OUTPUT_DIR).absolute() / f"{vowel_label}.csv"),
        ],
        capture_output=True,
        text=True,
    )

    # Print script output or errors
    print("stdout:", result.stdout)
    print("stderr:", result.stderr)

    # Optional: check if it failed
    if result.returncode != 0:
        print("❌ Script failed!")
