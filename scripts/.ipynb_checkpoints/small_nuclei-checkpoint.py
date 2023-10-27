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


tissue = 'WD-76845-097'


def fmt(x, pos):
    a, b = '{:.1e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'small_nuclei'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# read unfiltered single-cell data
data = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/sardana-097/output/' +
    'checkpoints/getSingleCellData.parquet'
    )

# crop
# ymin = 13200
# ymax = 13450
# xmin = 8325
# xmax = 8575

# ymin = 11338
# ymax = 11523
# xmin = 1845
# xmax = 2020

ymin = 10426
ymax = 10606
xmin = 1968
xmax = 2171

# read DNA1 image of selected tissue and crop to target size
dna = imread(
    '/Volumes/My Book/cylinter_input/sardana-097/' +
    f'tif/{tissue}.ome.tif', key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read segmentation outlines for target tissue sample and crop ROI
seg = imread(
    '/Volumes/My Book/cylinter_input/sardana-097/' +
    f'seg/{tissue}.ome.tif', key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]

# plot histogram
gate = (19, 27)
sns.set_style('whitegrid')
plt.hist(
    data['Area'],
    bins=20, color='k', range=(0, 250), density=False, lw=0.0
    )
plt.axvline(gate[1], c='tab:red')
plt.title(f'Sample {tissue}')
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
plt.imshow(dna_crop, alpha=0.5, cmap='Greys_r', vmin=400, vmax=40000)
plt.grid(False)

cbar = plt.colorbar(format=ticker.FuncFormatter(fmt))
# cbar = plt.colorbar()
cbar.set_alpha(1.0)
cbar.set_label('Hoechst per cell', labelpad=20, rotation=270)
cbar.draw_all()

plt.scatter(
    data_roi['X_centroid'],
    data_roi['Y_centroid'], s=4.5, lw=0.0, color='lime'
    )

plt.title(f'Sample {tissue}')
plt.savefig(
    os.path.join(
        save_dir,
        'sardana.png'), dpi=1000
        )

# save as a pdf, too
plt.savefig(
    os.path.join(
        save_dir,
        'sardana.pdf')
        )
plt.close('all')
