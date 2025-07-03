from pathlib import Path
import re
from random import sample
from typing import List, Optional

def sample_files(files: List[Path], n: int) -> List[Path]:
    return sample(files, min(n, len(files)))

def parse_snr_from_filename(filename: str) -> Optional[int]:
    match = re.search(r'snr(-?\d+)', filename.lower())
    return int(match.group(1)) if match else None 