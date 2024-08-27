from autologging import logged, traced
import iogeo.api
from riverpy.input import Geospatial
from plotea.api import plot_topo
import xarray

@logged
class Xarray:
  @classmethod
  def add_kilometres(cls, xarr, xcoord='x', ycoord='y'):
    xarr = xarr.assign_coords(xkm=(xarr[xcoord] / 1000.0))
    xarr = xarr.assign_coords(ykm=(xarr[ycoord] / 1000.0))
    return xarr
  @classmethod
  def get_extent(cls, xarr, unit='km'):
    if unit == 'km':
      extent=[xarr.xkm.min(), xarr.xkm.max(), xarr.ykm.min(), xarr.ykm.max()]
    else:
      extent=[xarr.x.min(), xarr.x.max(), xarr.y.min(), xarr.y.max()]
    return extent

@logged
class Topo(Geospatial):
  """
  Topography represented as xarray.

  """
  def read_xarr(self, source):
    if isinstance(source, xarray.DataArray):
      xarr = source
    elif isinstance(source, str):
      xarr = iogeo.api.read(source)
    else:
      raise NotImplementedError(f'Source type: {type(source)}')
    return xarr
  def read(self, source):
    self.xarr = self.read_xarr(source)
    self.xarr = Xarray.add_kilometres(self.xarr)
    self.extent = Xarray.get_extent(self.xarr, unit='km')
    self.arr = self.xarr.values.squeeze()
    return self.xarr
  def plot(self, *args, unit='km', **kwargs):
    extent = Xarray.get_extent(self.xarr, unit=unit)
    return plot_topo(self.arr, *args, extent=extent, **kwargs)
  