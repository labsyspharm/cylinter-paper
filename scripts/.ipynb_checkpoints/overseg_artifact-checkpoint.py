import os
import matplotlib.pyplot as plt
from tifffile import imread

core = '15'

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'overseg_artifact'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# crop coordinates
ymin = 1730
ymax = 1930
xmin = 1115
xmax = 1315

# read cycle1 DNA for selected tissue and crop to target size
dna = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    key=0
    )
dna_crop = dna[ymin:ymax, xmin:xmax]

# read segmentation outlines
seg = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif',
    key=0
    )
seg_crop = seg[ymin:ymax, xmin:xmax]


viewer = napari.view_image(
    dna, rgb=False, name=dna1
    )

viewer.add_image(
    seg, rgb=False, blending='additive',
    opacity=0.5, colormap='gray', visible=False,
    name='segmentation'
    )

napari.run()

file = open(os.path.join(save_dir, 'check_desktop.txt'), 'w')
file.write('Check for screen shot on desktop.')
file.close()
