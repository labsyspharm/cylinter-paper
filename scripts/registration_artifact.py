import os
import pandas as pd
import napari
from tifffile import imread
# import pygetwindow
# import pyautogui
# from PIL import Image

core = '64'
dna1 = f'DNA1:{core}'

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'alignment_artifact'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

markers = pd.read_csv(
    '/Volumes/My Book/cylinter_input/emit22_full/markers.csv'
    )

# read DNA1 channel
dna = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    key=0
    )

pcreb_channel_number = markers['channel_number'][
            markers['marker_name'] == 'pCREB']
pcreb = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    key=pcreb_channel_number.item() - 1
    )

ccnb1_channel_number = markers['channel_number'][
            markers['marker_name'] == 'CCNB1']
ccnb1 = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    key=ccnb1_channel_number.item() - 1
    )

ccne_channel_number = markers['channel_number'][
            markers['marker_name'] == 'CCNE']
ccne = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    key=ccne_channel_number.item() - 1
    )

# read segmentation outlines
seg = imread(
    f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif',
    key=0
    )


viewer = napari.view_image(
    dna, rgb=False, name=dna1
    )

viewer.add_image(
    seg, rgb=False, blending='additive',
    opacity=0.5, colormap='red', visible=False,
    name='segmentation'
    )

viewer.add_image(
    pcreb, rgb=False, blending='additive',
    opacity=1.0, colormap='green', visible=False,
    name='pCREB'
    )

viewer.add_image(
    ccnb1, rgb=False, blending='additive',
    opacity=1.0, colormap='green', visible=False,
    name='CCNB1'
    )

viewer.add_image(
    ccne, rgb=False, blending='additive',
    opacity=1.0, colormap='green', visible=False,
    name='CCNE'
    )

napari.run()

file = open(os.path.join(save_dir, 'check_desktop.txt'), 'w')
file.write('Check for screen shot on desktop.')
file.close()
