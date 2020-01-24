import pandas as pd


df = pd.read_csv('token2.csv')
#print(df.head(10))

from fuzzywuzzy import fuzz

def get_ratio(row):
    name = row['message']
    name1 = row['match']
    return fuzz.token_set_ratio(name, name1)

print(len(df[df.apply(get_ratio, axis=1) > 10]) / len(df))

import matplotlib.pyplot as plt
d1 = 'Depressive', 'Non-depressive'
d2 = [47.47,52.53]
colors = ['yellowgreen', 'lightcoral']
figureObject, axesObject = plt.subplots()
axesObject.pie(d2,



               autopct='%1.2f',

               startangle=90)

# Aspect ratio - equal means pie is a circle
patches, texts = plt.pie(d2, colors=colors, shadow=False, startangle=90)
plt.legend(patches, d1, loc="best")
axesObject.axis('equal')

plt.show()


