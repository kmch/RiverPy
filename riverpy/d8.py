"""
Tools for handling river flow directions in ESRI's D8 format, 
including an interface with Alex Lipp's `autocatchment` package.

"""
import numpy as np
import xarray as xr


# def topo
  # mg = ac.toolkit.load_topo(fname_out)

# def read_from_netcdf_file
  # ac.toolkit.load_d8
  # def load_d8(path: str) -> RasterModelGrid:
  #   """Reads a (generic) geospatial file containing ESRI D8
  #   flow directions (e.g., 0, 1, ..., 128) and calculates
  #   drainage network, assigning this to a landlab RasterModelGrid.

def extract_d8_subdomain(x1, x2, y1, y2, fname_d8_all, fname_d8,\
  pad_width=10, pad_value=0):
  """
  The output D8 file is readable both by `ac.toolkit.load_d8` 
  (i.e. `read_geo_file` that it calls) and by `snu.get_sample_graphs`.
  """


  # Extract, pad & save to file 
  data_array = xr.open_dataset(fname_d8_all)['z'].sel(x=slice(x1, x2), 
                                                      y=slice(y1, y2))
  data_array_vals = data_array.transpose('y', 'x').values
  data_array_vals[np.isnan(data_array_vals)] = -9999
  data_array_vals = np.flipud(data_array_vals)
  # Pad with zeroes
  data_array_vals =  np.pad(data_array_vals, pad_width, 'constant', 
                            constant_values=(pad_value))

  # Save to file (incl. padding in the header)
  dx = data_array.x[1] - data_array.x[0]
  dx = int(dx.values)
  ny = data_array.shape[1] + 2 * pad_width
  nx = data_array.shape[0] + 2 * pad_width
  newx1 = data_array.x.values[0] - pad_width * dx
  newx2 = data_array.x.values[-1] + pad_width * dx
  newy1 = data_array.y.values[0] - pad_width * dx
  newy2 = data_array.y.values[-1] + pad_width * dx
  header = f"ncols {ny}\n"
  header += f"nrows {nx}\n"
  header += f"xllcorner {newx1}\n"
  header += f"yllcorner {newy1}\n"
  header += f"cellsize {dx}\n"

  with open(fname_d8, 'w') as f:
    f.write(header)
    np.savetxt(f, data_array_vals, fmt='%d')
  data = np.loadtxt(fname_d8, skiprows=6)
  data = data[:-1, :-1] # invert yaxis?
  
  # Create coordinate arrays
  x_coords = np.linspace(newx1, newx2, data.shape[1])
  y_coords = np.linspace(newy1, newy2, data.shape[0])
  
  # Create xarray DataArray
  data = xr.DataArray(data, coords=[y_coords, x_coords], dims=['y', 'x'])

  return data





# Not used any more, but kept just in case:
def read_d8_header(file_path):
  with open(file_path, 'r') as file:
    header_lines = [next(file) for _ in range(6)]
  header_dict = {}
  for line in header_lines:
      line_parts = line.strip().split()
      if len(line_parts) == 2:
          key, value = line_parts
          header_dict[key.lower()] = float(value)

  xllcorner = header_dict.get('xllcorner', 0.0)
  yllcorner = header_dict.get('yllcorner', 0.0)
  cellsize = header_dict.get('cellsize', 1.0)
  ncols = int(header_dict.get('ncols', 0))
  nrows = int(header_dict.get('nrows', 0))

  x1 = xllcorner
  x2 = xllcorner + cellsize * ncols
  y1 = yllcorner
  y2 = yllcorner + cellsize * nrows

  dx = cellsize
  dy = cellsize
  nx = ncols
  ny = nrows

  return x1, x2, y1, y2, dx, dy, nx, ny


