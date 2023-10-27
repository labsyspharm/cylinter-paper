import pandas as pd

# import patient metadata
md = pd.read_csv('/Users/greg/Dropbox (HMS)/topacio/metadata/patient/metadata.csv')
# set Subject column as index
md.set_index('Subject', inplace=True)
# abbreviate not-evaluable BORs as 'ND'
md['Confirmed BOR'] = [
    'ND' if i in ['Not Done', 'Not Evaluable'] else
    i for i in md['Confirmed BOR']
]

# abbreviate BRCA statuses
md['BRCA'] = [
    i.split('-')[1] if '-' in i else
    i for i in md['BRCA']
]

md.index = [i.split('_')[1].lstrip('0') for i in md.index]

md['binary_response'] = [
    'response' if i in ['CR', 'PR'] else 'progression' if i in ['SD', 'PD'] else 
    i for i in md['Confirmed BOR']
]

myorder = [
    '39', '40', '89', '73', '81', '104', '127', '66', '55', '70', '110', '31', '48', '53', '96',
    '125', '91', '83', '86', '95', '124', '128', '106', '152', '115'
]
md = md[md.index.isin(myorder)]
md_sorted = md.reindex(myorder)
