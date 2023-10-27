import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from itertools import combinations
from scipy.stats import pearsonr

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/last_minute_figures'
    '/CRC_collagen_rugplot')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# SARDANA
df = pd.read_parquet(
    '/Volumes/T7 Shield/cylinter_input/sardana-097/output/'
    'checkpoints/clustering.parquet'
    )
channels2 = ['CollagenIV_647']

handles = []
for channel in channels2:

    print(channel)

    cutoff = np.percentile(df[channel], q=0.01)

    plt.hist(
        df[channel], bins=300, density=True, histtype='step',
        linewidth=5.0, alpha=1.0
        )

    plt.hist(
        cluster[channel], bins=50, density=True, histtype='step',
        linewidth=5.0, alpha=1.0
        )

    plt.xlim([cutoff, 1.0])

    y_max = plt.gca().get_ylim()[1]

    # # add and annotate rug plot
    # for i in examples:
    #     plt.plot(
    #         cluster[channel][cluster.index == i], y_max*0.015,
    #         marker='|', color='k',
    #         )
    #
    #     label = example_dict[i]
    #     plt.annotate(
    #         f'{label}',
    #         (cluster[channel][cluster.index == i], y_max*0.03),
    #         textcoords='offset points', xytext=(0, 0), ha='center',
    #         fontsize=6
    #         )

    plt.savefig(os.path.join(save_dir, f'{channel}_crop.pdf'))
    plt.close('all')
