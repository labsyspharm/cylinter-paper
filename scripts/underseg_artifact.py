import os
import napari
import tifffile
import zarr
import pandas as pd


def single_channel_pyramid(tiff_path, channel):
    target_filepath = tiff_path
    tiff = tifffile.TiffFile(target_filepath)
    pyramid = [zarr.open(s[channel].aszarr()) for s in tiff.series[0].levels]

    return pyramid


core = '17'

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/' +
    'underseg_artifact'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = pd.read_parquet(
    '/Volumes/My Book/cylinter_input/emit22_full/' +
    'output/checkpoints/dnaIntensityCutoff.parquet')

sample_data = data[data['Sample'] == core]
centroids = sample_data[['Y_centroid', 'X_centroid']][
    (sample_data['Area'] > 400)
    ]

# read cycle1 DNA for selected tissue and crop to target size
dna = single_channel_pyramid(
    f'/Volumes/My Book/cylinter_input/emit22_full/tif/{core}.ome.tif',
    channel=0
    )

# read segmentation outlines
seg = single_channel_pyramid(
    f'/Volumes/My Book/cylinter_input/emit22_full/seg/{core}.ome.tif',
    channel=0
    )


viewer = napari.view_image(
    dna, rgb=False, name='DNA1'
    )

viewer.add_image(
    seg, rgb=False, blending='additive',
    opacity=0.5, colormap='green', visible=False,
    name='segmentation'
    )

viewer.add_points(
    centroids, name='DNA area',
    face_color='white',
    edge_color='k',
    edge_width=0.0, size=1.0
    )

napari.run()

file = open(os.path.join(save_dir, 'check_desktop.txt'), 'w')
file.write('Check for screen shot on desktop.')
file.close()
