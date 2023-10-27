import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from tifffile import imread

tissue = '840063_0110-median_r_50'
vmin = 700
vmax = 65535


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'bright_nuclei'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read unfiltered single-cell data
data = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/topacio_cellring/output/' +
    'checkpoints/getSingleCellData.parquet'
    )

# full image
ymin = 0
ymax = 1000000
xmin = 0
xmax = 1000000

# crop
# ymin = 12800
# ymax = 13400
# xmin = 17900
# xmax = 18500

# read DNA1 image of selected tissue and crop to target size
dna = imread(
    '/Volumes/My Book/cylinter_input/topacio_cellring/' +
    f'tif/{tissue}.ome.tif', key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read segmentation outlines for target tissue sample and crop ROI
seg = imread(
    '/Volumes/My Book/cylinter_input/topacio_cellring/' +
    f'seg/{tissue}.ome.tif', key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]

# plot cropped image channels, segmentation outlines, and analysis rectangle
# plt.imshow(seg_crop, alpha=1.0, cmap='Greys_r')
plt.imshow(dna_crop, alpha=1.0, cmap='magma', vmin=vmin, vmax=vmax)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
# cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

plt.savefig(
    os.path.join(
        save_dir,
        'tissue_0110.png'), dpi=1000
        )
plt.close('all')

# isolate data for cells within ROI bounds (original coordinates)
gate = (55000, 65535)
data_roi = data[['Y_centroid', 'X_centroid', 'DNA0_cellRingMask']][
    (data['Sample'] == tissue) &
    (data['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (data['Y_centroid'].between(ymin, ymax, inclusive=True))
    ]

sns.set_style('whitegrid')
plt.hist(
    data_roi['DNA0_cellRingMask'],
    bins=40, color='k', density=False, lw=0.0
    )
plt.axvline(gate[0], c='tab:red')
plt.title(f'Sample {tissue}')
plt.xlabel('DNA1')
plt.ylabel('cell count')
plt.savefig(
    os.path.join(
        save_dir,
        'gated_hist.pdf')
        )
plt.close('all')

gated_centroids = data_roi[['Y_centroid', 'X_centroid']][
    (data_roi['DNA0_cellRingMask'].between(gate[0], gate[1], inclusive=True))
    ]

# transform x, y coordinates of gated cells within ROI bounds
gated_centroids['X_centroid'] = data_roi['X_centroid']-xmin
gated_centroids['Y_centroid'] = data_roi['Y_centroid']-ymin

# plot channel images and segmentation outlines
# plt.imshow(seg_crop, alpha=1.0, cmap='Greys_r')
plt.imshow(dna_crop, alpha=1.0, cmap='magma', vmin=vmin, vmax=vmax)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
# cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

# overlay cell centorids
plt.scatter(
    gated_centroids['X_centroid'],
    gated_centroids['Y_centroid'], s=0.4, lw=0.0, color='lime'
    )
plt.title(f'Sample {tissue}')
plt.savefig(
    os.path.join(
        save_dir,
        'bright_nuclei.png'), dpi=1000
        )

# save as a pdf, too
plt.savefig(
    os.path.join(
        save_dir,
        'bright_nuclei.pdf')
        )
plt.close('all')
