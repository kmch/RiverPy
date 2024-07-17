from autologging import logged, traced

def process_data_samples(data_samples, drainage, params):
    clusters = data_samples.cluster(max_dist=100, min_clusters=2)
    means = clusters.export_statistic('mean')
    sigms = clusters.export_statistic('std')
    for samples in [means, sigms]:
      samples.snap_to_drainage(drainage, thresh=params.thresh)
      samples.create_sample_graph(drainage.fname)
      for sample_id, new_coords in params.relocate_dict.items():
        samples.relocate_sample(sample_id=sample_id, new_coords=new_coords)
      samples.snap_to_drainage(drainage, thresh=params.thresh)
      samples.create_sample_graph(drainage.fname)
      samples.rename_based_on_topology(use_letters=params.use_letters)
    clusters.update_ids(means.df)
    return means, sigms, clusters
