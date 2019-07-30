import numpy as np
import pandas as pd

df = pd.read_csv("../datasets/ufcData.csv")

one_hot = pd.get_dummies(df['winner'])
df = df.drop('winner', axis=1)

df = df.select_dtypes(include=['int', 'float'])
df = df.join(one_hot)

df = df.dropna(how='all', axis='columns')
df = df.fillna(df._get_numeric_data().mean())

df.to_csv('cleanUfc.csv', sep=',')
