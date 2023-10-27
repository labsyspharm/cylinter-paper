import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from tifffile import imread

core = '12'
channels = [f'DNA1:{core}']
vmax = 18000


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'dim_nuclei'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read unfiltered single-cell data
data = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/emit22_full/output/' +
    'checkpoints/getSingleCellData.parquet'
    )

# full image
# ymin = 0
# ymax = 1000000
# xmin = 0
# xmax = 1000000

# crop
ymin = 1775
ymax = 1925
xmin = 1250
xmax = 1400

# read DNA1 image of selected tissue and crop to target size
dna = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif', key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read segmentation outlines for target tissue sample and crop ROI
seg = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif', key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]

# plot cropped image channels, segmentation outlines, and analysis rectangle
plt.imshow(seg_crop, alpha=1.0, cmap='Greys_r')
plt.imshow(dna_crop, alpha=0.6, cmap='magma', vmax=vmax)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
# cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

plt.savefig(
    os.path.join(
        save_dir,
        'core_12.png'), dpi=1000
        )
plt.close('all')

# isolate data for cells within ROI bounds (original coordinates)
gate = (0, 3500)
data_roi = data[['Y_centroid', 'X_centroid', 'DNA1_cellMask']][
    (data['Sample'] == core) &
    (data['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (data['Y_centroid'].between(ymin, ymax, inclusive=True))
    ]

data_gated = data[['Y_centroid', 'X_centroid', 'DNA1_cellMask']][
    (data['DNA1_cellMask'].between(gate[0], gate[1], inclusive=True))
    ]

sns.set_style('whitegrid')
plt.hist(
    data['DNA1_cellMask'][data['Sample'] == core],
    bins=40, color='k', density=False, lw=0.0
    )
plt.axvline(gate[1], c='tab:red')
plt.title(f'Core {core}')
plt.xlabel('DNA1')
plt.ylabel('cell count')
plt.savefig(
    os.path.join(
        save_dir,
        'gated_hist.pdf')
        )
plt.close('all')

# transform x, y coordinates of cells within ROI bounds
data_gated['X_centroid'] = data_roi['X_centroid']-xmin
data_gated['Y_centroid'] = data_roi['Y_centroid']-ymin

# plot channel images and segmentation outlines
plt.imshow(seg_crop, alpha=1.0, cmap='Greys_r')
plt.imshow(dna_crop, alpha=0.6, cmap='magma', vmax=vmax)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
# cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

# overlay cell centorids
plt.scatter(
    data_gated['X_centroid'],
    data_gated['Y_centroid'], s=4.5, lw=0.0, color='lime'
    )
plt.title(f'Core {core}')
plt.savefig(
    os.path.join(
        save_dir,
        'dim_nuclei.png'), dpi=1000
        )

# save as a pdf, too
plt.savefig(
    os.path.join(
        save_dir,
        'dim_nuclei.pdf')
        )
plt.close('all')
