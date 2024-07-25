from autologging import logged, traced

# import sys
# sys.path.append('..') # to access plotea
from plotea.mpl2d import Shade #TODO

from typing import Tuple, Dict, List #, Final, Iterator, , Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes._axes import Axes
import pandas as pd

# import networkx as nx
import xarray as xr
import geopandas as gpd
import rasterio

@traced
@logged
def plot_sweep_of_regularizer_strength(sample_network, element_data,
    min_: float,
    max_: float,
    trial_num: float,
    plot = True) -> [list, list]:
    """
    THIS IS A FUNCTION COPIED FROM fasterunmixer SLIGHTLY MODIFIED.

    Plot a sweep of regularization strengths and their impact on roughness and data misfit.

    Args:
        sample_network (nx.DiGraph): The network of sample sites along the drainage, with associated data.
        element_data (ElementData): Dictionary of element data.
        min_ (float): The minimum exponent for the logspace range of regularization strengths to try.
        max_ (float): The maximum exponent for the logspace range of regularization strengths to try.
        trial_num (float): The number of regularization strengths to try within the specified range.

    Note:
        The function performs a sweep of regularization strengths within a specified logspace range and plots their
        impact on the roughness and data misfit of the sample network. For each regularization strength value, it
        solves the sample network problem using the specified solver ("ecos") and the corresponding regularization
        strength. It then calculates the roughness and data misfit values using the network's `get_roughness()` and
        `get_misfit()` methods, respectively.

        The roughness and data misfit values are plotted as a scatter plot, with the regularization strength value
        displayed as text next to each point. The x-axis represents the roughness values, and the y-axis represents the
        data misfit values.

        The function also prints the roughness and data misfit values for each regularization strength value.

        Finally, the function displays the scatter plot with appropriate axis labels.

    Returns:
        None
    """
    vals = np.logspace(min_, max_, num=trial_num)  # regularizer strengths to try
    rough, misf = [], []
    for val in vals:
        # plot_sweep_of_regularizer_strength._log.info(20 * "_")
        plot_sweep_of_regularizer_strength._log.info("Trying regularizer strength: 10^ %s" %round(np.log10(val), 3))
        _ = sample_network.solve(element_data, solver="ecos", regularization_strength=val)
        roughness = sample_network.get_roughness()
        misfit = sample_network.get_misfit()
        rough.append(roughness)
        misf.append(misfit)
        # plot_sweep_of_regularizer_strength._log.info("Roughness:", np.round(roughness, 4))
        # plot_sweep_of_regularizer_strength._log.info("Data misfit:", np.round(misfit, 4))
        if plot:
          plt.scatter(roughness, misfit, c="grey")
          plt.text(roughness, misfit, str(round(np.log10(val), 3)))
          plt.xlabel("Roughness")
          plt.ylabel("Data misfit")
    return vals, rough, misf




# def snap_a_point_to_drainage(x, y, drainage, thresh=1e7):  
#   coords = np.array([x,y])
#   snappd = ac.autosampler.snap_to_drainage(drainage.mg, coords, thresh)
#   assert len(coords) == len(snappd)
#   # self.__log.info('len coords vs. snappd', len(coords), len(snappd))
#   self.df['x'] , self.df['y']  = ac.toolkit.model_xy_to_geographic_coords(
#       (snappd[:, 0], snappd[:, 1]), drainage.mg)

def gauss(x, mu=0, sigma=1):
  """
  Unnormalized Gaussian distribution.
  
  Parameters
  ----------
  
  Returns
  -------
  y : type(x)
    Gaussian evaluated at x. 
  
  Notes
  -----
  Some people use alpha (1/e point)
  instead of the sigma (standard deviation)
  to define the width of the Gaussian. 
  They are related through: alpha = sigma * sqrt(2)
  
  """
  return np.exp(-((x - mu)**2) / (2 * sigma**2))
def gaussND(dims, centers, radius, **kwargs):
  """
  This should be vectorized for speed.
  
  """
  coords = np.indices(dims)
  A = np.zeros(dims)
  for center in centers: 
    distance = np.sqrt((coords[0] - center[0])**2 + (coords[1]-center[1])**2 + (coords[2]-center[2])**2) 
    a = gauss(distance, 0, radius)
    A += a
  return A

def log10(df):
  return df.applymap(lambda x: np.log10(x) if pd.notnull(x) else x)
def log(df): # alias
  return log10(df)
def round2accuracy(to_round, accuracy):
    return np.round(to_round / accuracy) * accuracy
def sort(datastruct, flip=True):
  
  if type(datastruct) == pd.DataFrame:
    ascending = True if flip else False
    df = datastruct
    means = df.mean()
    sorted_columns = means.sort_values(ascending=ascending).index
    df_sorted = df[sorted_columns]
    return df_sorted
  elif type(datastruct) == dict:
    return {k: datastruct[k] for k in sorted(datastruct)}
  else:
    raise TypeError('Unsupported type: %s' % type(datastruct))



class SampleNode:
  """
  For duck-typing of pyfastunmix.SampleNode
  """
  def __init__(self):
    self.name = None
    self.x = None
    self.y = None
    self.area = None
    self.total_upstream_area = None
    self.downstream_node = None
    self.my_export_rate = None
    self.my_flux = None
    self.my_total_flux = None
    self.my_total_tracer_flux = None
    self.my_tracer_flux = None
    self.my_tracer_value = None
    self.upstream_nodes = None



def exclude_thames_trunk(samples: pd.DataFrame) -> pd.DataFrame:
  return samples[samples['Water_Category_Details'] != 'Thames']
def exclude_all_trunk(samples: pd.DataFrame) -> pd.DataFrame:
  # labels copied from Excel in three steps
  string = """O2 Arena 2 3
  O2 Arena 2 2
  O2 Arena 2 1
  O2 Arena 1 3
  O2 Arena 1 2
  O2 Arena 1 1
  Bermondsey 3
  Bermondsey 2
  Bermondsey 1
  Tower Bridge 3
  Tower Bridge 2
  Tower Bridge 1
  Southwark Bridge 2
  Southwark Bridge 1
  Millenium Bridge 3
  Millenium Bridge 2
  Millenium Bridge 1
  Blackfrias Bridge 3
  Blackfrias Bridge 2
  Blackfrias Bridge 1
  Gabriels Wharf 2
  Gabriels Wharf 1
  Westminster Bridge 3
  Westminster Bridge 2
  Westminster Bridge 1
  H. Parliament3
  H. Parliament2
  H. Parliament1
  Lambeth Bridge 3
  Lambeth Bridge 2
  Lambeth Bridge 1
  Vauxhall Tower 3
  Vauxhall Tower 2
  Vauxhall Tower 1
  US Embassy 3
  US Embassy 2
  US Embassy 1
  Battersea Park 3
  Battersea Park 2
  Battersea Park 1
  Wandsworth Park 3
  Wandsworth Park 2
  Wandsworth Park 1
  Putney Bridge 3
  Putney Bridge 2
  Putney Bridge 1
  Putney Rowing 3
  Putney Rowing 2
  Putney Rowing 1
  Kew Bridge 3
  Kew Bridge 2
  Kew Bridge 1
  Kew Palace 3
  Kew Palace 2
  Kew Palace 1
  Richmond 3
  Richmond 2
  Richmond 1
  Teddington 2
  Teddington 1
  Kingston 3
  Kingston 2
  Kingston 1
  Upstream Erith 3
  Upstream Erith 2
  Upstream Erith 1
  Downstream Crossness 3
  Downstream Crossness 2
  Downstream Crossness 1
  Crossness Outfall 3
  Crossness Outfall 2
  Crossness Outfall 1
  Upstream Crossness 3
  Upstream Crossness 2
  Upstream Crossness 1
  Downstream Beckton 3
  Downstream Beckton 2
  Downstream Beckton 1 
  Beckton Outfall 3
  Beckton Outfall 2
  Beckton Outfall 1
  Erith 3
  Erith 2
  Erith 1
  Junction
  Thames
  WWTP"""
  labels = string.splitlines()
  for label in labels:
    label = label.strip()
    samples = samples[samples['Water_Category_Details'] != label]
  return samples
def find_period_with_most_data(samples: pd.DataFrame, \
  chemical: str, time_window) -> Tuple[int, pd.Timestamp, Dict[str, int], pd.DataFrame]:
  
  samples = exclude_thames_trunk(samples)

  df = select_single_chemical_subset(samples, chemical)
  
  # Exclude NaNs
  df = df.dropna(subset=[chemical])

  # print('After dropna: ', len(df))

  # Convert the 'Sampling_Date' column to pandas datetime format
  df['Sampling_Date'] = pd.to_datetime(df['Sampling_Date'])

  # Sort the DataFrame by the 'Sampling_Date' column
  df = df.sort_values('Sampling_Date')

  # Set the 'Sampling_Date' column as the index
  df = df.set_index('Sampling_Date')

  # Initialize variables to store the maximum count and the corresponding period
  max_count = 0
  max_period = None

  dates = {}
  counts = []
  best_period = None
  # Iterate over each date in the DataFrame
  for date in df.index:

      # Define the start and end dates for the period
      start_date = date
      end_date = start_date + pd.DateOffset(days=time_window-1)

      # Filter the DataFrame for the current period
      period_df = df.loc[start_date:end_date]

      # Count the number of samples in the period
      count = len(period_df)
      
      dates[date] = count
      
      # Check if the current period has a higher count than the previous maximum
      if count > max_count:
          max_count = count
          max_period = (start_date, end_date)
          best_period = period_df

  # Print the maximum count and the corresponding period
  # print("Maximum count:", max_count)
  # print("Period:", max_period)

  return max_count, max_period, dates, best_period
def select_single_chemical_subset(samples: pd.DataFrame, chemical: str):
  key = chemical
  non_chems = ['x_bng', 'y_bng', 'Sample Code', 'Location', 'Water_Category_Details', 'Year', 'Sampling_Date', 'Latitude', 'Longitude']
  chems = samples.drop(non_chems, axis=1).columns.tolist()
  return samples.drop([i for i in chems if i != key], axis=1)
def select_subset_drain(rivers, river_name):
  rivers = rivers[rivers[col1].str.contains(name)]
def select_subset(samples, rivers, name):
    col1 = 'WB_NAME'
    col2 = 'Water_Category_Details'
    if rivers is not None:
      rivers = rivers[rivers[col1].str.contains(name)]
    samples = samples[samples[col2].str.contains(name)]
    return rivers, samples
def split_dataset_into_chemicals(samples: pd.DataFrame) -> dict[str, pd.DataFrame]:
  non_chems = ['x_bng', 'y_bng', 'Sample Code', 'Location', 'Water_Category_Details', 'Year', 'Sampling_Date', 'Latitude', 'Longitude']
  chems = samples.drop(non_chems, axis=1).columns.tolist()
  di = {}
  for key in chems:
    di[key] = samples.drop([i for i in chems if i != key], axis=1)
  return di  


def set_lims(ax, river_name):
  rn = river_name
  if rn == 'lea':
    ax.set_xlim((530000,545000))
    ax.set_ylim((180000,192600))
  elif rn == 'hogsmill':
    ax.set_xlim((515000,523000))
    ax.set_ylim((162010,171000))
  elif rn == 'beverley':
    ax.set_xlim((519000,530000))
    ax.set_ylim((162010,177500))
  elif rn == 'wandle':
    ax.set_xlim((523500,532500))
    ax.set_ylim((163500,177500))
  elif rn == 'guc':
    ax.set_xlim((504300,520000))
    ax.set_ylim((175000,183000))
  else:
    raise ValueError('Unknown river name (%s)' % river_name)
  return ax
def set_lims_around_sample(ax, df, sample_id, pad):
  sdf = df[df['Sample.Code'] == sample_id]
  x, y = sdf['x_coordinate'], sdf['y_coordinate']
  x, y = float(x.iloc[0]), float(y.iloc[0])
  x1, x2 = x - pad, x + pad
  y1, y2 = y - pad, y + pad
  ax.set_xlim(x1,x2)
  ax.set_ylim(y1,y2)
  return ax



def plot_data_coverage_over_time(dates: dict, \
  max_count: int, time_window: int, chemical: str):
  plt.subplots(figsize=(15,10))

  # Convert the dictionary to separate lists for x and y values
  x = list(dates.keys())
  y = list(dates.values())

  # Plot the data
  plt.plot(x, y, '.-')

  # Customize the plot
  plt.xlabel('First day of a %s-day time-window' % time_window)
  plt.ylabel('No. of non-NaN samples in the time-window')
  plt.title('%s - max count: %s' % (chemical, max_count))
  plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
  return plt.gca()
def plot_samples_in_BNG_color(samples: pd.DataFrame, c: str, ax: Axes, **kwargs) -> Axes:
  xcol = 'x_bng'
  ycol = 'y_bng'
  sc1 = ax.plot(samples[xcol].values, samples[ycol].values, 'o', color=c, **kwargs)

  return plt.gca()
def plot_samples_in_BNG(samples: pd.DataFrame, chem: str, ax: Axes, \
  col='chem', nans=False, **kwargs) -> Axes:
  xcol = 'x_bng'
  ycol = 'y_bng'
  if not nans:
    kwargs['edgecolor'] = kwargs.get('edgecolor', 'none')
  

  if col == 'chem':
    if nans:
      # kwargs['edgecolor'] = kwargs.get('edgecolor', 'r')
      # kwargs['facecolor'] = kwargs.get('facecolor', 'none')
      sc1 = ax.scatter(samples[xcol].values, samples[ycol].values, \
                       facecolor='none', edgecolor='r', **kwargs)
    else:
      kwargs['cmap'] = kwargs.get('cmap', 'viridis')
      sc1 = ax.scatter(samples[xcol].values, samples[ycol].values, 
                      c=np.log10(samples[chem].values), **kwargs)
      cbar = colorbar(sc1, ax)
      cbar.set_label('log10(concentration)')
  elif col == 'date':
    import matplotlib.dates as mdates
    samples = samples.copy()
    samples.reset_index(inplace=True)
    dates1 = pd.to_datetime(samples['Sampling_Date'])
    dates1 = mdates.date2num(dates1)
    if nans:
      # print(';kdhaf')
      # kwargs['edgecolor'] = kwargs.get('edgecolor', 'r')
      # kwargs['facecolor'] = kwargs.get('facecolor', 'none')
      sc1 = ax.scatter(samples[xcol].values, samples[ycol].values, 
                    edgecolor='r', facecolor='none', **kwargs)
    # kwargs['cmap'] = kwargs.get('cmap', 'tab20c')
    else:
      sc1 = ax.scatter(samples[xcol].values, samples[ycol].values, 
                    c=dates1, **kwargs)
    cbar = colorbar(sc1, ax, format=mdates.DateFormatter('%Y-%m-%d'))
    cbar.set_label('Sampling date')
  else:
    raise TypeError()

  return plt.gca()
def plot_samples_in_BNG_dates(samples: pd.DataFrame, chem: str, ax: Axes, **kwargs) \
  -> Axes:
  xcol = 'x_bng'
  ycol = 'y_bng'
  kwargs['edgecolor'] = kwargs.get('edgecolor', 'none')
  kwargs['cmap'] = kwargs.get('cmap', 'viridis')
  sc1 = ax.scatter(samples[xcol].values, samples[ycol].values, 
                    c=np.log10(samples[chem].values), **kwargs)
  cbar = colorbar(sc1, ax)
  cbar.set_label('log10(concentration)')
  return plt.gca()
def plot_topography(X, Y, Z, **kwargs):
  fig, ax = plt.subplots(figsize=[12,8])
  shader = Shade()
  extent=[X.min(), X.max(), Y.min(), Y.max()]
  shader.plot(Z.T, cmap='Greys_r', extent=extent, aspect='equal',
              azdeg=45, altdeg=45, label='Topography, m a.s.l.', **kwargs)
  ax.set_xlabel('X - British National Grid, m')
  ax.set_ylabel('Y - British National Grid, m')
  ax.invert_yaxis()
  return fig, ax

