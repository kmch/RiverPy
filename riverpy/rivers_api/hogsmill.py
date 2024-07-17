from dataclasses import dataclass, field
from riverpy.rivers_api.base import ParamsBase
from typing import Dict, List, Optional

@dataclass
class ParamsHogsmill(ParamsBase):
  """    
  Usage
  >>> params = ParamsHogsmill()
  >>> print(params)
  >>> print(params.relocate_dict)
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