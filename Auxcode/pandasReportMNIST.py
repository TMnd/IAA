import pandas as pd
# from pandas_profiling import ProfileReport

df = pd.read_csv("../dados/mnist_train_WithSymbols_Balance_v2.csv")

output = {}

for index, row in df.iterrows():
    if index == 0:
        print(len(row))
    key = row['label']
    if key in output:
        output[key] = output.get(key)+1
    else:
        output[key] = 1

print(output)