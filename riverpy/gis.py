from autologging import logged
import iogeo
from iogeo.xr import Xarr
import numpy as np
from plotea.api import plot_topo
from riverpy.hydro import Drain
from riverpy.input import Geospatial
import xarray as xr

@logged
class DEM(Geospatial):
  """
  Topography represented as xarray.

  """
  def compute_drainage(self, fname):
    return Drain().compute_from_dem_netcdf(fname)
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
  