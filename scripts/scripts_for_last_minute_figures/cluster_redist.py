import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

save_dir = (
    '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/last_minute_figures'
    '/cluster_redistribution'
    )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

raw = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/' +
    'output_raw/checkpoints/clustering.parquet'
    )

clean = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/' +
    'output_orig/checkpoints/clustering.parquet'
    )

clean['handle'] = clean['Sample'] + '_' + clean['CellID'].astype(str)

cois = [415, 404, 424, 174, 272]

plt.scatter(
    clean['emb1'], clean['emb2'], c='gainsboro', s=0.05, linewidth=0.0
    )
for clus in cois:
    test = raw[['Sample', 'CellID']][raw['cluster_2d'] == clus]
    ids = test['Sample'] + '_' + test['CellID'].astype(str)
    myshit = clean[clean['handle'].isin(ids)]
    plt.scatter(
        myshit['emb1'], myshit['emb2'], s=0.1, linewidth=0.0, label=clus
    )

font = font_manager.FontProperties(family='Arial')
plt.legend(prop=font, bbox_to_anchor=[1.18, 1.01], markerscale=30)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cluster_redist.png'), dpi=800)
plt.close('all')
