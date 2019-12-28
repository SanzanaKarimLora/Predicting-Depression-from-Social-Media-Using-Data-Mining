import pandas as pd
import re

df = pd.read_csv('token_tweet.csv')
print(df.head(10))

dfsad = pd.read_csv('sad.csv')
print(dfsad.head(4))

from fuzzywuzzy import fuzz
print(fuzz.ratio('Deluxe Room', 'Deluxe King Room'))

print(fuzz.token_sort_ratio('Deluxe Room, 1 King Bed', 'Deluxe King Room'))

def get_ratio(row):
    name = df['message']
    name1 = dfsad['message']
    return fuzz.token_set_ratio(name, name1)

print(len(df[df.apply(get_ratio, axis=1) > 10]) / len(df))
