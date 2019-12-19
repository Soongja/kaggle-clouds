import pandas as pd
import numpy as np

# train = pd.read_csv('../data/train.csv')
df = pd.read_csv('../submissions/submission_191118c.csv')
df2 = pd.read_csv('../submissions/submission_191118d.csv') # size coverage
df3 = pd.read_csv('../submissions/submission_191118e.csv') # max val
df4 = pd.read_csv('../submissions/submission_191118f.csv') # max val size coverage
# df5 = pd.read_csv('../submissions/submission_191115b.csv')


def count_predicted_classes(df):
    counts = []
    for i in range(4):
        count = 0
        for j in range(len(df) // 4):
            pred = df.loc[4*j + i]['EncodedPixels']
            if not pd.isnull(pred):
                count += 1

        counts.append(count)

    return counts

# print([(c / 5546) for c in count_predicted_classes(train)])
# print([(c / 5546) for c in count_predicted_classes(df)])
# print([(c / 5546) for c in count_predicted_classes(df2)])
# print([(c / 5546) for c in count_predicted_classes(df3)])
# print([(c / 5546) for c in count_predicted_classes(df4)])
# print([(c / 5546) for c in count_predicted_classes(df5)])
# print(count_predicted_classes(train))
print(count_predicted_classes(df))
print(count_predicted_classes(df2))
print(count_predicted_classes(df3))
print(count_predicted_classes(df4))
# print(count_predicted_classes(df5))
