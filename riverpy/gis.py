from autologging import logged, traced
import iogeo.api
import numpy as np
from plotea.api import plot_topo
from riverpy.input import Geospatial
import xarray as xr
from riverpy.xr import Xarr


@logged
class DEM(Geospatial):
  """
  Topography represented as xarray.

  """
  def read_xarr(self, source):
    if isinstance(source, xr.DataArray):
      xarr = source
    elif isinstance(source, str):
      xarr = iogeo.api.read(source)
    else:
      raise NotImplementedError(f'Source type: {type(source)}')
    return xarr
  def read(self, source):
    self.xarr = self.read_xarr(source)
    self.xarr = Xarr.add_kilometres(self.xarr)
    self.extent = Xarr.get_extent(self.xarr, unit='km')
    self.arr = self.xarr.values.squeeze()
    return self.xarr
  def plot(self, *args, unit='km', **kwargs):
    extent = Xarr.get_extent(self.xarr, unit=unit)
    return plot_topo(self.arr, *args, extent=extent, **kwargs)
  