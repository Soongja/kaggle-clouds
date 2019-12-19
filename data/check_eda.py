import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('eda.csv')

classes = ['Fish', 'Flower', 'Gravel', 'Sugar']

# for cls in classes:
#     df_1 = df.loc[df['numComponents'] == 1]
#     df_cls = df_1.loc[df_1['fname'].apply(lambda x: x.split('_')[1]) == (cls + '.png')]
#     minareas = df_cls['minComponentArea'].values
#     minareas = [int(minarea / 16) for minarea in minareas]
#     print(np.min(minareas))

save_df = pd.DataFrame(columns=classes)

for cls in classes:
    df_hasmask = df.loc[df['numComponents'] >= 1]
    df_cls = df_hasmask.loc[df_hasmask['fname'].apply(lambda x: x.split('_')[1]) == (cls + '.png')]

    coverages = df_cls['coverage'].values
    minCompareas = df_cls['minComponentArea'].values
    # print(np.mean(coverages))

    coverages.sort()
    minCompareas.sort()
    # for i in range(len(coverages)):
    #     print(i+1, coverages[i])
    for i in range(len(minCompareas)):
        print(i+1, minCompareas[i])
    print('#############################################################################')
    # plt.hist(coverages, bins=100)
    # plt.show()