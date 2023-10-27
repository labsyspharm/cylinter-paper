import os

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches

from tifffile import imread

from skimage.color import gray2rgb
from skimage import img_as_float


core = '84'


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'large_nuclei'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read unfiltered single-cell data
data = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/emit22_full/output/' +
    'checkpoints/getSingleCellData.parquet'
    )
data = data[data['Sample'] == core]

# crop
ymin = 1785
ymax = 1985
xmin = 2030
xmax = 2230

# read DNA1 image of selected core and crop to target size
dna = imread(
    '/Volumes/My Book/cylinter_input/emit22_full/' +
    f'tif/{core}.ome.tif', key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read segmentation outlines for target core sample and crop ROI
seg = imread(
    '/Volumes/My Book/cylinter_input/emit22_full/' +
    f'seg/{core}.ome.tif', key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]

# plot histogram
gate = (245, 443)
gate = (200, 443)
sns.set_style('whitegrid')
plt.hist(
    data['Area'],
    bins=35, color='k', density=False, lw=0.0
    )
plt.axvline(gate[0], c='tab:red')
plt.axvline(gate[1], c='tab:red')
plt.title(f'Sample {core}')
plt.xlabel('Segmentation Area')
plt.ylabel('cell count')
plt.savefig(
    os.path.join(
        save_dir,
        'gated_hist.pdf')
        )
plt.close('all')

# plot cropped image window
data_roi = data[['Y_centroid', 'X_centroid', 'Area']][
    (data['X_centroid'].between(xmin, xmax, inclusive=True)) &
    (data['Y_centroid'].between(ymin, ymax, inclusive=True)) &
    (data['Area'].between(gate[0], gate[1], inclusive=True))
    ]
data_roi['X_centroid'] = data_roi['X_centroid']-xmin
data_roi['Y_centroid'] = data_roi['Y_centroid']-ymin

seg_crop = gray2rgb(img_as_float(seg_crop)) * (1, 0, 0)
plt.imshow(seg_crop, alpha=1.0)
plt.imshow(dna_crop, alpha=0.4, cmap='Greys_r', vmin=500, vmax=20000)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
# cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

plt.scatter(
    data_roi['X_centroid'],
    data_roi['Y_centroid'], s=6.5, lw=0.0, color='lime'
    )

plt.title(f'Sample {core}')
plt.savefig(
    os.path.join(
        save_dir,
        f'{core}.png'), dpi=1000
        )

# save as a pdf, too
plt.savefig(
    os.path.join(
        save_dir,
        f'{core}.pdf')
        )
plt.close('all')
