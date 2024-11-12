from iogeo.api import read
from iogeo.nc import NetCDF
import numpy as np
import xarray as xr
from autocatchments.toolkit import load_d8, load_topo

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

class FlowDirs:
  def compute_drainage(self, fname):
    return Drain().compute_from_d8_netcdf(fname)
  def extract(self, x1, x2, y1, y2, fname_inp, fname_out, data_array="z", 
              pad_width=10, pad_value=0, save=False):
    """
    Extracts a subdomain from a D8 flow direction file, pads it, 
    and saves as a NetCDF file readable by xarray.

    Notes
    -----
    It needs padding.

    """
    xarr = NetCDF.extract(x1, x2, y1, y2, fname_inp, fname_out, 
                          data_array, pad_width, pad_value, save)
    return xarr
  def read(self, fname):
    return read(fname)

class Drain:
  def compute_from_dem_netcdf(self, fname):
    grid = load_topo(fname) # FIXME: prohibitively slow for thames_elev.nc
    xarr = RasterModelGrid.to_xarray(grid)
    return xarr
  def compute_from_d8_netcdf(self, fname):
    grid = load_d8(fname)
    xarr = RasterModelGrid.to_xarray(grid)
    return xarr
  def save(self):
    pass
