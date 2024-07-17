from riverpy.rivers_api.hogsmill import *

def Params(river: str):
  if river.lower() == 'hogsmill':
    return ParamsHogsmill()
  else:
    raise NotImplementedError(f"No parameters available for river: {river}")