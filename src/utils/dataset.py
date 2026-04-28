
from pathlib import Path
import re


def get_all_speakers(data_path):
    """
    Returns speaker folder names strictly matching s1, s2, ... (no suffixes)
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_path}")

    speakers = []
    pattern = re.compile(r"^s\d+$")  # STRICT match (no suffixes)

    for item in data_path.iterdir():
        if item.is_dir() and pattern.match(item.name):
            speakers.append(item.name)

    return sorted(speakers)


def parse_speakers(speaker_args, all_speakers):
    if not speaker_args:
        return all_speakers

    selected = []

    for item in speaker_args:
        if "-" in item:
            try:
                start, end = item.split("-")
                start_idx = int(start[1:])
                end_idx = int(end[1:])

                for i in range(start_idx, end_idx + 1):
                    selected.append(f"s{i}")
            except Exception:
                print(f"[WARNING] Invalid range format: {item}")
        else:
            selected.append(item)

    # Remove duplicates
    selected = list(set(selected))

    # Keep only valid speakers
    invalid = [s for s in selected if s not in all_speakers]
    if invalid:
        print(f"[WARNING] Invalid speakers ignored: {invalid}")

    selected = [s for s in selected if s in all_speakers]

    return sorted(selected)

