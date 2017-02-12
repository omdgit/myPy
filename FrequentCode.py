# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 14:45:40 2017

@author: omdgit
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from urllib.request import urlopen
from collections import OrderedDict

print("Python version: %s" % sys.version)
print("Pandas version: %s" % pd.__version__)
print("Numpy version: %s" % np.__version__)
print("Matplotlib version: %s" % matplotlib.__version__)

url_meta = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
            'pima-indians-diabetes/pima-indians-diabetes.names')
url_data = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
            'pima-indians-diabetes/pima-indians-diabetes.data')
colnames = ['times_pregnant', 'plasma_clucose_conc', 'diastol_blood_pressure',
            'triceps_skin_fold_thickness', 'two_hr_serum_insulin',
            'body_mass_index', 'diab_pedigree_func', 'age_years',
            'class']
            
# information on data set
with urlopen(url_meta) as description:
    for line in description:
        print(line)
        
df = pd.read_csv(url_data, names=colnames)
df.info()
df.head()

# proc format equivalent
bins = [0, 1, 2, 3, 17]
pregnancies = ['None', '1', '2', '>2']
df['pregnancies'] = pd.cut(df['times_pregnant'], bins, 
                             labels=pregnancies, right=False)
df[['times_pregnant', 'pregnancies']].head()
df.groupby('pregnancies').size()

aggregations = OrderedDict([
    ('Minimum', 'min'),
    ('Maximum', 'max'),
    ('Mean', 'mean'),
    ('StDev', 'std'),
    ('Median', 'median'),
    ('N', 'count')
])

df.groupby('pregnancies')['plasma_clucose_conc'].agg(aggregations)
