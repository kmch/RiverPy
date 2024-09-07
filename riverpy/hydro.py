from iogeo.api import read
from iogeo.nc import NetCDF
# from iogeo.xr import Xarr
import numpy as np
from riverpy.gis import RasterModelGrid
import xarray as xr
from autocatchments.toolkit import load_d8, load_topo

class FlowDirs:
  def compute_drainage(self, fname):
    return Drain().compute_from_d8_netcdf(fname)
  def extract(self, x1, x2, y1, y2, fname_inp, fname_out, pad_width=10, pad_value=0):
    """
    Extracts a subdomain from a D8 flow direction file, pads it, 
    and saves as a NetCDF file readable by xarray.

    Notes
    -----
    It needs padding.
    
    """
    xarr = NetCDF.extract(x1, x2, y1, y2, fname_inp, fname_out, pad_width, pad_value)
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
