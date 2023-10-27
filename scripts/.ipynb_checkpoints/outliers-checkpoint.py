import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
import napari
from umap import UMAP
import hdbscan
from joblib import Memory

core = '68'
cd63_gate = 3.0
# cd63_gate = 2.7
# core = '39'
# cd63_gate = 3.0
# core = '23'
# cd63_gate = 3.0

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'outliers'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read unfiltered single-cell data
data = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/emit22_full/output/' +
    'checkpoints/getSingleCellData.parquet'
    )
core_data = data[data['Sample'] == core].copy()

markers = pd.read_csv(
    '/Volumes/My Book/cylinter_input/emit22_full/markers.csv'
    )

aoi = [
    i for i in list(markers['marker_name']) if not
    any(x in i for x in ['DNA', 'IgG', 'CD56', 'CD13', 'pAUR',
        'CCNE', 'CDKN2A', 'PCNA_1', 'CDKN1B_2'])
        ]

# isolate columns for antibodies of interest
clustering_cols = [
    i for i in core_data.columns if i.split('_cellMask')[0] in aoi
    ]

# log-transform antibody column data
transformed_data = np.log10(core_data[clustering_cols])

# update core_data with log-transformed values
core_data.update(transformed_data)

# drop immunomarkers outliers from core_data
core_data_gated = core_data[core_data['CD63_cellMask'] < cd63_gate]

# full image
ymin = 0
ymax = 1000000
xmin = 0
xmax = 1000000

###############################################################################
# read cycle1 DNA for selected tissue and crop to target size
dna = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read CD63 channel for selected tissue and crop to target size
cd63 = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    key=18
    )
cd63_crop = cd63[ymin:ymax, xmin:xmax]

# read segmentation outlines
seg = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif',
    key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]
###############################################################################

plt.scatter(core_data['CD63_cellMask'], core_data['Area'], c='k', s=0.5)
plt.title(f'Core {core}')
plt.xlabel('CD63')
plt.ylabel('Area')
plt.savefig(
    os.path.join(
        save_dir,
        f'cd63_{core}_before_gating.pdf')
        )
plt.close('all')

plt.scatter(
    core_data_gated['CD63_cellMask'], core_data_gated['Area'], c='k', s=0.5
    )
plt.title(f'Core {core}')
plt.xlabel('CD63')
plt.ylabel('Area')
plt.savefig(
    os.path.join(
        save_dir,
        f'cd63_{core}_afted_gating.pdf')
        )
plt.close('all')
###############################################################################

outlier_centroids = core_data[['Y_centroid', 'X_centroid']][
    core_data['CD63_cellMask'] > cd63_gate]

cd63_intensity = core_data['CD63_cellMask'][
    core_data['CD63_cellMask'] > cd63_gate].values

point_properties = {
    'CD63_intensity': cd63_intensity
    }

viewer = napari.view_image(
    dna, opacity=0.5, rgb=False, name='DNA1'
    )

viewer.add_image(
    seg, rgb=False, blending='additive',
    opacity=0.5, colormap='green', visible=False,
    name='segmentation'
    )

viewer.add_image(
    cd63, rgb=False, blending='additive',
    colormap='magma', visible=True,
    contrast_limits=(0, 10289),
    name='CD63'
    )

viewer.add_points(
    outlier_centroids, name='CD63 outliers',
    properties=point_properties,
    face_color='lime',
    # face_colormap='viridis',
    # edge_color='viridis',
    edge_width=0.0, size=4.0
    )

napari.run()
###############################################################################

# cluster before removing outliers
print('Performing "before" embedding...')
cldata1 = core_data[clustering_cols].copy()
embedding1 = UMAP(random_state=5, min_dist=0.05).fit_transform(cldata1)
cldata1['emb1'] = embedding1[:, 0]
cldata1['emb2'] = embedding1[:, 1]

clustering1 = hdbscan.HDBSCAN(min_cluster_size=15).fit(
        cldata1[['emb1', 'emb2']]
        )
cldata1['cluster'] = clustering1.labels_

# cluster after removing outliers
print('Performing "after" embedding...')
cldata2 = core_data_gated[clustering_cols].copy()
embedding2 = UMAP(random_state=5, min_dist=0.05).fit_transform(cldata2)
cldata2['emb1'] = embedding2[:, 0]
cldata2['emb2'] = embedding2[:, 1]

clustering2 = hdbscan.HDBSCAN(min_cluster_size=15).fit(
        cldata2[['emb1', 'emb2']]
        )
cldata2['cluster'] = clustering2.labels_

fig, ax = plt.subplots(2)
ax[0].scatter(cldata1['emb1'], cldata1['emb2'], c=cldata1['CD63_cellMask'], cmap='viridis', s=0.3)
ax[0].set_aspect('equal')
ax[1].scatter(cldata2['emb1'], cldata2['emb2'], c=cldata2['CD63_cellMask'], cmap='viridis', s=0.3)
ax[1].set_aspect('equal')
plt.savefig(
    os.path.join(
        save_dir,
        f'embedding_{core}.pdf')
        )
plt.close('all')

# colors = {-1: '#000000', 0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c',
#           3: '#d62728', 4: '#9467bd', 5: '#8c564b', 6: '#e377c2',
#           7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf'}
# cldata['color_cluster'] = [colors[i] for i in cldata['cluster']]
#################################################################

file = open(os.path.join(save_dir, 'check_desktop.txt'), 'w')
file.write('Check for screen shot on desktop.')
file.close()
