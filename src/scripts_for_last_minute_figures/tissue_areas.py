import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
from ome_types import from_tiff

paths = {
    'topacio70':('/Volumes/T7 Shield/cylinter_input/TOPACIO_FINAL/tif/840047_0070.ome.tif', 1000),
    'topacio39':('/Volumes/T7 Shield/cylinter_input/TOPACIO_FINAL/tif/840003_0039.ome.tif', 1000),
    'topacio40':('/Volumes/T7 Shield/cylinter_input/TOPACIO_FINAL/tif/840004_0040.ome.tif', 700),
    'topacio125':('/Volumes/T7 Shield/cylinter_input/TOPACIO_FINAL/tif/840072_0125.ome.tif', 1200),
    'topacio128':('/Volumes/T7 Shield/cylinter_input/TOPACIO_FINAL/tif/840153_0128.ome.tif', 600),
    'sardana':('/Volumes/T7 Shield/cylinter_input/sardana-097/tif/WD-76845-097.ome.tif', 1000),
    'codex1':('/Volumes/My Book/cylinter_input/CODEX/tif/sample_1.ome.tif', 200),
    'codex2':('/Volumes/My Book/cylinter_input/CODEX/tif/sample_2.ome.tif', 1300),
    'emit':('/Volumes/My Book/T7_overflow/cylinter_input/emit22_full/tif/1.ome.tif', 200),
    'mihc': ('/Volumes/My Book/cylinter_input/mIHC/tif/sample_1.ome.tif', 30),
}

for name, data in paths.items():
    path = data[0]
    cutoff = data[1]
    ome = from_tiff(path)
    im = imread(path, key=0)
    binary = (im > cutoff).astype(int)
    plt.imshow(binary)
    #plt.show()
    pixel_size = round(ome.images[0].pixels.physical_size_x, 2)
    count_ones = np.count_nonzero(binary == 1)
    print(
        f'Pixel size of {name} is {pixel_size} with an estimated area of {count_ones} sq. pixels'
    )
