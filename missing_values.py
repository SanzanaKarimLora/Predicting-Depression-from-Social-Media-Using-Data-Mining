# importing pandas module
import pandas as pd
from importlib import reload
import sys
if sys.version[0] == '2':
     reload(sys)
     sys.setdefaultencoding("ISO-8859-1")
# making data frame from csv file

missing = pd.read_csv("preprocess_tweet.csv",encoding = 'ISO-8859-1')
print(missing.head())

print(missing.isnull())
print(missing.isnull().sum())
print(missing["message"].isnull().sum())
print(missing["label"].isnull().sum())

missing["message"].fillna("NaN", inplace = True)
missing["label"].fillna("NaN", inplace = True)

missing_values = ["n/a", "na", "--"]
missing = pd.read_csv("preprocess_tweet.csv",encoding = 'ISO-8859-1', na_values = missing_values)


#missing=missing.fillna("null ")
modified = missing.dropna()
print(modified.isnull().sum())
modified.to_csv('moditweet.csv',index=False)

'''''
print(missing.isnull())
print(missing.isnull().sum())
print(missing["message"].isnull().sum())
print(missing["label"].isnull().sum())
'''