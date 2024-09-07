from iogeo.api import read
import numpy as np
from riverpy.gis import RasterModelGrid
from riverpy.xr import Xarr
import xarray as xr
from autocatchments.toolkit import load_d8

class NetCDF4:
  pass

class FlowDirs:
  def extract(self, x1, x2, y1, y2, fname_inp, fname_out, pad_width=10, pad_value=0):
    """
    Extracts a subdomain from a D8 flow direction file, pads it, 
    and saves as a NetCDF file readable by xarray.
    """
    xarr = xr.open_dataset(fname_inp)['z'].sel(x=slice(x1, x2), y=slice(y1, y2))
    xarr = Xarr.pad(xarr, pad_width, pad_value)
    dataset = xr.Dataset({"z": xarr})
    dataset.to_netcdf(fname_out, format='NETCDF4')
    return xarr
  def read(self, fname):
    return read(fname)

class Drain:
  def get_from_d8_netcdf(self, fname):
    grid = load_d8(fname)
    extent = RasterModelGrid.get_extent(grid)
    arr = grid.at_node["drainage_area"].reshape(grid.shape)
    xarr = Xarr.create_from_ndarray(arr, extent)
    return xarr

  def save(self):
    pass