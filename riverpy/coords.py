import pyproj

def transform_coords(inp_x, inp_y, inp_coord_system='latlon', out_coord_system='bng'):
  transformer = create_coord_transformer(inp_coord_system, out_coord_system)
  out_x, out_y = transformer.transform(inp_x, inp_y)
  return out_x, out_y

def create_coord_transformer(inp='latlon', out='bng'):
  coord_system_ID = {
    'bng'         : 'EPSG:27700',   # British National Grid
    'latlon'      : 'EPSG:4326',    # WGS84 (latitude and longitude)
    'openstreet'  : 'EPSG:3857', 
  }

  inp_ = pyproj.CRS(coord_system_ID[inp])  
  out_ = pyproj.CRS(coord_system_ID[out])  

  transformer = pyproj.Transformer.from_crs(inp_, out_, always_xy=True)
  return transformer

def convert_xy_to_ij(x, y, extent, dx, dy):
  x1, x2, y1, y2 = extent
  i = int((x - x1) / dx)
  j = int((y2 - y) / dy) # yes!! because the array is numbered from top y
  return i, j

