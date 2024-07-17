import matplotlib.pyplot as plt
import numpy as np
from plotea.mpl2d import *


class PlotGeospatial(AxesFormatter):
  def __init__(self, domain=None, topo=None, drain=None, rivers=None, catch=None, \
    csos=None, wtws=None, samples=None, graph=None, gauges=None):
    self.domain = domain
    self.extent = self.domain.extent
    self.topo = topo
    self.drain = drain 
    self.rivers = rivers
    self.catch = catch
    self.csos = csos
    self.wtws = wtws
    self.samples = samples
    self.graph = graph
    self.gauges = gauges
  def plot(self, ax=None, model=None, data=None, zoom=None, **kwargs):
    ax = get_ax(ax)
    self.plot_bckg(ax, **kwargs)
    
    if zoom is not None:
      x, y, pad = zoom
      self._zoom(x, y, pad, ax=ax)
    
    self._adjust(ax, **kwargs)
    # self._convert_ticks_m2km(ax)
    # self._set_xylabels()
    # ax = remove_every_nth_tick_label(ax=ax, n=2, axis='x', start=1)
    return ax
  def plot_bckg(self, ax=None, domain=True, topo=True, drain=True, rivers=True, \
     catch=True, csos=True, wtws=True, samples=True, graph=True, gauges=False,\
      **kwargs):
    ax = get_ax(ax)
    if topo:
      self.topo.plot(ax=ax, cbar=0, zorder=1)
    if catch:
      self.catch.plot(ax=ax, zorder=3)
    if drain:
      self.drain.plot(ax=ax, zorder=5)      
    if rivers:
      self.rivers.plot(ax=ax, zorder=7)
    if graph:
      self.graph.plot_spatial(ax=ax, width=200, c='w', zorder=9)            
    if wtws:
      self.wtws.plot(ax=ax, zorder=11)
    if csos:
      self.csos.plot(ax=ax, zorder=13)
    if gauges:
      self.gauges.plot(ax=ax, s=400, c='w', zorder=15)
    if samples:
      self.samples.plot(ax=ax, s=200, norm=None, column='id', cmap='rainbow', marker='o', annotate='id', 
      annotate_kw=dict(xytext=(0,-2)), zorder=17)

    return ax  
  def plot_df(self, xcol='x', ycol='y', ccol=None, acol=None):
    #     s='duration', s_factor=0.1, cbar=True, cbar_label=None, **kwargs):
    #     ax = get_ax(ax)
    #     # if s == 'duration':
    #     #   s = df[self.tcol] * s_factor
    #     sc = ax.scatter(df['Easting'], df['Northing'], marker=marker, c=c, s=s, **kwargs)  
    #     if cbar:
    #       colorbar_flexible(sc, ax, cbar_label)
    #     self._set_xylabels()
    #     return ax
    #   def plot_df(self, df, xcol, ycol, ccol=None, scol=None, acol=None, \
    #     extent=None, **kwargs):
    #     # df = self._restrict_to_extent(self.df, extent) if extent is not None else 
    #     return
    pass
  def _get_extent_from_df(self):
    """
    Works only for object having associated DataFrames.
    """
    x1 = self.df['x'].min()
    x2 = self.df['x'].max()
    y1 = self.df['y'].min()
    y2 = self.df['y'].max()
    return [x1, x2, y1, y2]
  def _get_extent_zoom(self, x, y, pad=1000):
    x1, x2 = x - pad, x + pad
    y1, y2 = y - pad, y + pad
    return x1, x2, y1, y2
  def _zoom(self, x, y, pad=1000, ax=None):
    if ax is None:
      ax = plt.gca()
    x1, x2, y1, y2 = self._get_extent_zoom(x, y, pad)
    self.extent_zoom = [x1, x2, y1, y2]
    ax.set_xlim(x1, x2)
    ax.set_ylim(y1, y2)
    return ax


# Figures ------------------------------------------------------------------------------

def plot_multichem(chem):
  def add_subfigure_label(ax, label, pos='tl', fontsize=20, c='w'):
      y2 = 0.98
      if pos == 'tl':
        pos = [0.03, y2]
      elif pos == 'tr':
        pos = [y2, y2]
      elif pos == 'br':
        pos = [y2, 0.08]
      elif pos == 'bl':
        pos = [0.03, 0.08]
      else:
        pass
      ax.text(*pos, label, transform=ax.transAxes, c=c,
              fontsize=fontsize, va='top')
      return ax
    kws = dict(samples=0, graph=0, catch=1, rivers=0, wtws=0, drain=1, csos=0)
    kwsd = dict(column=chem, vmin=cmin, vmax=cmax, rotate_ylabels=1, 
                s=400, marker='o', zorder=15,
                annotate='id', annotate_kw=dict(c='w', xytext=(0,-3)),)
    fig = plt.figure(figsize=(13,7))
    gs = gridspec.GridSpec(10,4, width_ratios=[2,1.4,1,0.5])
    # --------------------------------
    ax = fig.add_subplot(gs[:,0])
    ax = ps.plot(ax=ax, **kws)
    ax = dobs.plot(ax=ax, **kwsd)
    Domain(*Geospatial().get_extent_zoom(*zoom1)).plot(ax=ax, c='w')
    Domain(*Geospatial().get_extent_zoom(*zoom2)).plot(ax=ax, c='w')
    add_subfigure_label(ax, 'a', pos=[0.03, 0.99])
    # --------------------------------
    ax = fig.add_subplot(gs[:4,1])
    ax = ps.plot(ax=ax, zoom=zoom1, labels=0, ticklabels=1, **kws)
    ax = dobs.plot(ax=ax, labels=0, ticklabels=1, **kwsd)
    #     plt.xticks([525000,527000], [525,527])
    #     plt.yticks([173000,175000], [173,175])
    add_subfigure_label(ax, 'b', pos='tl')
    # --------------------------------
    ax = fig.add_subplot(gs[5:-1,1])
    ax = ps.plot(ax=ax, zoom=zoom2, labels=0, ticklabels=1, **kws)
    ax = dobs.plot(ax=ax, labels=0, ticklabels=1, **kwsd)
    #     plt.xticks([529000,531000], [529,531])
    #     plt.yticks([165000,167000], [165,167])
    add_subfigure_label(ax, 'c', pos='tl')
    # --------------------------------
    ax = fig.add_subplot(gs[-1,1])
    ax.axis('off')
    cax = ax.inset_axes([0.08, 0., .85, .3])
    fig.colorbar(dobs.sc, ax=ax, cax=cax, location='bottom', orientation='horizontal',
                    label='Concentration (ng/l)')
    from matplotlib.ticker import ScalarFormatter
    cax.xaxis.set_major_formatter(ScalarFormatter())
    # --------------------------------
    ax = fig.add_subplot(gs[:,2])
    clusters.plot_boxplot(chem, ax=ax)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.grid(None)
    add_subfigure_label(ax, 'd', pos=[.06, .99], c='k')
    # --------------------------------
    i = 1
    for cid, clu in clusters.dict.items():
        ax = fig.add_subplot(gs[-i,3])
        if i == 10:
            add_subfigure_label(ax, 'e', pos=[-.25, .999], c='k')
        xtick_labels = True if i == 1 else False
        ytick_labels = True if i == 5 else False
        ylabel = True if i == 5 else False    
        clu.plot_conc(chem, dates, ax=ax, xtick_labels=xtick_labels, decimate_xticklabels=2,
                          ytick_labels=ytick_labels, ylim=(.1,3.), ylabel=ylabel)
        i += 1


# def plot_data_multichem(chem: str, figsize=(13,7), width_ratios=[2,1,1,0.5]):
  #   fig = plt.figure(figsize=figsize)
  #   gs = gridspec.GridSpec(10, 4, width_ratios=width_ratios)
  #   # a) Full map --------------------------------
  #   ax = fig.add_subplot(gs[:,0])
  #   # b) Zoom-ins & colorbar --------------------------------
  #   ax = fig.add_subplot(gs[ :4,1])  # zoom-in #1
  #   ax = fig.add_subplot(gs[4:-1,1]) # zoom-in #2
  #   ax = fig.add_subplot(gs[-1,1])   # colorbar
  #   # c) Boxplots --------------------------------
  #   ax = fig.add_subplot(gs[:,2]) 
  #   # d) Clusters broken-down --------------------------------
  #   for i in range(10):
  #     ax = fig.add_subplot(gs[-i,3])


