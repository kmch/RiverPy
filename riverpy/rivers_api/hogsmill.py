from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ParamsHogsmill:
  """    
  Usage
  >>>ph = ParamsHogsmill()
  >>>print(ph)
  >>>print(ph.relocate_dict)
  """
  thresh: float = 1e6
  use_letters: bool = True
  relocate_dict: Dict[int, List[Optional[float]]] = field(default_factory=lambda: {
    6: [520500, 164300],
    9: [None, 163400],
  })




# DomainHogsmill

# CatchHogsmill

# TopoHogsmill?