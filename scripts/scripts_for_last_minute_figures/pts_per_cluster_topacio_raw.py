import pandas as pd

# save_dir = (
#     '/Users/greg/Dropbox (HMS)/Baker_QC_2021/script_output/last_minute_figures'
#     '/pts_per_cluster_topacio_raw')
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

df = pd.read_parquet(
    '/Users/greg/Dropbox (HMS)/topacio/cylinter_output/TOPACIO_FINAL/'
    'output_raw/checkpoints/clustering.parquet'
)
df = df[df['cluster_2d'] != -1]

mydict = {}
for name, group in df.groupby(['cluster_2d']):
    mydict[name[0]] = (group['Sample'].nunique() / df['Sample'].nunique()) * 100

min_key = min(mydict, key=lambda k: mydict[k])
max_key = max(mydict, key=lambda k: mydict[k])

print(
    f'Cluster with smallest specimen diversity is {min_key} with {mydict[min_key]}% of specimens.'
)
print(
    f'Cluster with largest specimen diversity is {max_key} with {mydict[max_key]}% of specimens.'
)

single_sample_clusters = [key for key, value in mydict.items() if value == 4.0]
print(f'Single sample clusters are {single_sample_clusters}.')

sorted_dict = dict(sorted(mydict.items(), key=lambda item: item[1]))
