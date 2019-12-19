import pandas as pd
import numpy as np

df = pd.read_csv('submission_5.csv')
print(df)

counts = []
for i in range(4):
    count = 0
    for j in range(1801):
        pred = df.loc[4*j + i]['EncodedPixels']
        if not pd.isnull(pred):
            count += 1

    counts.append(count)

print(counts)