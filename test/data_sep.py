import pandas as pd

df = pd.read_csv("data_test.csv", header=0)

size = 89282

for i in range(10):
    start = size*i
    end = size*(i+1)
    if i==5:
        df.iloc[start:][:].to_csv("data_test_" + str(i + 1) + ".csv")
    else:
        df.iloc[start:end][:].to_csv("data_test_"+str(i+1)+".csv")