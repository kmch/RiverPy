from autologging import logged #, traced
import numpy as np
import xarray as xr

@logged
class Xarr:
  @classmethod
  def add_kilometres(cls, xarr:xr.DataArray, xcoord='x', ycoord='y'):
    xarr = xarr.assign_coords(xkm=(xarr[xcoord] / 1000.0))
    xarr = xarr.assign_coords(ykm=(xarr[ycoord] / 1000.0))
    return xarr
  @classmethod
  def get_extent(cls, xarr:xr.DataArray, unit='km'):
    if unit == 'km':
      extent=[xarr.xkm.min(), xarr.xkm.max(), xarr.ykm.min(), xarr.ykm.max()]
    else:
      extent=[xarr.x.min(), xarr.x.max(), xarr.y.min(), xarr.y.max()]
    return extent
  @classmethod
  def pad(cls, xarr:xr.DataArray, pad_width, pad_value):
    # Pad the array with the specified value
    xarr_vals = xarr.values
    xarr_vals = np.pad(xarr_vals, pad_width, 'constant', constant_values=(pad_value))
    
    # Calculate new coordinates after padding
    dx = xarr.x[1] - xarr.x[0]
    ny, nx = xarr_vals.shape
    newx1 = xarr.x.values[0] - pad_width * dx
    newx2 = xarr.x.values[-1] + pad_width * dx
    newy1 = xarr.y.values[0] - pad_width * dx
    newy2 = xarr.y.values[-1] + pad_width * dx
    
    # Create new coordinate arrays
    x_coords = np.linspace(newx1, newx2, nx)
    y_coords = np.linspace(newy1, newy2, ny)
    
    xarr = xr.DataArray(xarr_vals, coords=[y_coords, x_coords], dims=['y', 'x'])
    return xarr
