from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ParamsBase:
    """Base class with required parameters."""
    thresh: float
    use_letters: bool
    relocate_dict: Dict[int, List[Optional[float]]]

