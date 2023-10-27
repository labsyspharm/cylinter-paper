# on transfer:
# rsync -avP /n/standby/hms/hits/lsp/collaborations/lsp-data/cycif-production/57-bc-topacio-guerriero/57-bc-topacio-guerriero/raw/* --include raw/*.rcpnl --exclude "*.*" /n/scratch3/users/g/gjb15/topacio_tiles/

# on o2:
# srun --pty -p interactive --mem 500M -t 0-06:00 /bin/bash
# module load gcc/9.2.0
# module load python/3.9.14
# virtualenv ~/venvs/topacio_tiles
# source ~/venvs/topacio_tiles/bin/activate
# pip install --upgrade pip
# pip install ashlar
# python ~/scripts/ashlar/preview_slide.py

import warnings
import os
import sys
import numpy as np
import skimage.transform
import skimage.util
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.text as mtext
from ashlar import reg, utils


for e, (root, dirs, files) in enumerate(
        os.walk('/n/scratch3/users/g/gjb15/topacio_tiles/', topdown=True)
):
    if not (e == 0) and not (root.endswith('raw')):

        files_path = os.path.join(root, dirs[0])
        os.chdir(files_path)

        for file in os.listdir(files_path):

            filename, file_ext = os.path.splitext(file)

            if file_ext == '.rcpnl':
                for ch in [0, 1, 2, 3]:

                    print(
                        f'WORKING ON CHANNEL {ch} of' +
                        f'{os.path.join(files_path, file)}...'
                    )
                    reader = reg.BioformatsReader(os.path.join(files_path, file))
                    metadata = reader.metadata

                    resolution_scale = 1 / 10
                    positions = (
                        (metadata.positions - metadata.origin) * resolution_scale
                    )
                    pmax = (positions + metadata.size * resolution_scale).max(axis=0)
                    mshape = (pmax + 0.5).astype(int)
                    mosaic = np.zeros(mshape, dtype=np.uint16)

                    total = reader.metadata.num_images
                    for i in range(total):
                        sys.stdout.write("\rLoading %d/%d" % (i + 1, total))
                        sys.stdout.flush()
                        img = reader.read(c=ch, series=i)
                        img = skimage.transform.rescale(
                            img, resolution_scale, anti_aliasing=False
                        )
                        img = skimage.img_as_uint(img)

                        # log transform
                        intensity_scale = 65535 / np.log(65535)
                        img = (
                            (np.log(np.maximum(img, 1)) * intensity_scale)
                            .astype(np.uint16)
                        )

                        # Round position so paste skips expensive subpix shift
                        pos = np.round(positions[i])
                        utils.paste(mosaic, img, pos, np.maximum)
                    print()

                    ax = plt.gca()

                    plt.imshow(X=mosaic, axes=ax, extent=(0, pmax[1], pmax[0], 0))

                    h, w = metadata.size * resolution_scale
                    for i, (x, y) in enumerate(np.fliplr(positions)):

                        # show tile bounds
                        rect = mpatches.Rectangle((x, y), w, h, color='black', fill=False)
                        ax.add_patch(rect)

                    #     # show tile numbers
                    #     xc = x + w / 2
                    #     yc = y + h / 2
                    #     circle = mpatches.Circle(
                    #        (xc, yc), w / 5, color='salmon', alpha=0.5
                    #        )
                    #     text = mtext.Text(
                    #        xc, yc, str(i), color='k', size=10,
                    #        ha='center', va='center'
                    #        )
                    #     ax.add_patch(circle)
                    #     ax.add_artist(text)

                    plt.savefig(f'{filename}_{ch}.pdf')
        print()
