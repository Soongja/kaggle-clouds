import pandas as pd
import numpy as np


classes = ['Fish', 'Flower', 'Gravel', 'Sugar']
dirs = ['fold1_ensemble2', 'fold1_ensemble3', 'seg_fold3_ensemble2', 'seg_fold3_ensemble3']

for c in classes:
    print('-------------------- %s --------------------' % c)
    print('  th   size   cov   dice')
    for d in dirs:
        df = pd.read_csv('%s/%s.csv' % (d, c))
        print(df.head())
        # df = df.astype({"min_size": int})
        # best = df.loc[0].values
        # print('%.3f  %d  %.2f  %.4f  %s' % (best[0], best[1], best[2], best[3], d))
