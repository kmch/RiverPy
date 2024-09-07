from autologging import logged
import iogeo
from iogeo.xr import Xarr
import numpy as np
from plotea.api import plot_topo
from riverpy.hydro import Drain
from riverpy.input import Geospatial
import xarray as xr

class RasterModelGrid:
  @classmethod
  def get_extent(cls, grid):
    ny, nx = grid.shape
    dx = grid.dx
    dy = grid.dy
    x1, y1 = grid.xy_of_lower_left

    # Calculate extent
    x2 = x1 + (nx - 1) * dx
    y2 = y1 - (ny - 1) * dy

    return [x1, x2, y1, y2]
  @classmethod
  def to_xarray(cls, grid):
    extent = cls.get_extent(grid)
    arr = grid.at_node["drainage_area"].reshape(grid.shape)
    xarr = Xarr.create_from_ndarray(arr, extent)
    return xarr

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
  