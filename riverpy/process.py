
def process_hogsmill():
  river = 'hogsmill'
  thresh = 1e6
  use_letters = True
  relocate_dict = {
        'B': [520500, 164300],
        'C': [None, 163400],
      }
  dataset = DatasetICL()
  data_samples = dataset.select_subset(river)
  clusters = data_samples.cluster(max_dist=100, min_clusters=2)
  means = clusters.export_statistic('mean')
  sigms = clusters.export_statistic('std')

def prepare_data():
    data_samples = dataset.select_subset(river)
    clusters = data_samples.cluster(max_dist=max_dist, min_clusters=min_clusters)
    means = clusters.export_statistic('mean')
    sigms = clusters.export_statistic('std')
    means.process(drainage, **kwargs)
    sigms.process(drainage, **kwargs)
    clusters.update_ids(means.df)
    acq = Acquisition(means)

def process_data():
    fname_d8 = drainage.fname
    self.snap_to_drainage(drainage, thresh=thresh)
    self.create_sample_graph(fname_d8)
    self.rename_based_on_topology(use_letters=use_letters)
    for sample_id, new_coords in relocate_dict.items():
      self.relocate_sample(sample_id=sample_id, new_coords=new_coords)
    self.snap_to_drainage(drainage, thresh=thresh)
    self.create_sample_graph(fname_d8)      
    # self.rename_based_on_topology(use_letters=use_letters) #?  

