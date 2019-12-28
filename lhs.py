import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest

def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens

text = 'The devil went down to Georgia'
print('The shingles (tokens) are:', preprocess(text))

#Number of Permutations
permutations = 128

#Number of Recommendations to return
num_recommendations = 1


def get_forest(data, perms):
    start_time = time.time()

    minhash = []

    for text in data['message']:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
        forest = MinHashLSHForest(num_perm=perms)

    for i, m in enumerate(minhash):
     forest.add(i, m)

    forest.index()

    print('It took %s seconds to build forest.' % (time.time() - start_time))
    return forest


def predict(text, database, perms, num_results, forest):
    start_time = time.time()

    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))

    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None  # if your query is empty, return none

    result = database.iloc[idx_array]['label']

    print('It took %s seconds to query forest.' % (time.time() - start_time))

    return result

db = pd.read_csv('moditweet.csv')
db['message'] = db['message']
forest = get_forest(db, permutations)

num_recommendations = 5
title = 'Using a neural net to instantiate a deformable model'
result = predict(title, db, permutations, num_recommendations, forest)
print('\n Top Recommendation(s) is(are) \n', result)

'''
happy=np.array(['happy'])
import csv
a='good'     #String that you want to search
with open("moditweet.csv") as f_obj:
    reader = csv.reader(f_obj, delimiter=',')
    for line in reader:      #Iterates through the rows of your csv
        print(line)          #line here refers to a row in the csv
        if a in line:      #If the string you want to search is in the row
            print("String found in first row of csv")
        else:
            print("not found")
        break

'''
'''
import csv

f = open('moditweet.csv')
itm = dict()
v = 0
r = csv.reader(f)
l = ("sad", "hopeless", "upset", "lost", "want to be alone","depress")
#m = ("happy", "good", "well")
sum1=0

for row in r:
    v = row[0]
    if v in itm.keys():
        cnt = itm.get(v)
        for i in l:
            if (i in row[1]):
                cnt = cnt + 1;
                itm[v] = cnt;


    else:
        cnt = 0
        for i in l:
            if (i in row[1]):
                cnt = cnt + 1;
                itm[v] = cnt;
        sum1 = sum1+cnt

print("Sad text:")
for j in itm.keys():
    print(j, itm[j])

print(sum1)

f.close()



f = open('moditweet.csv')
itm = dict()
v = 0
r = csv.reader(f)
l = ("not happy", "not good", "not well")
#m = ("happy", "good", "well")
sum2=0
for row in r:
    v = row[0]
    if v in itm.keys():
        cnt = itm.get(v)
        for i in l:
            if (i in row[1]):
                cnt = cnt + 1;
                itm[v] = cnt;


    else:
        cnt = 0
        for i in l:
            if (i in row[1]):
                cnt = cnt + 1;
                itm[v] = cnt;
        sum2=sum2+cnt
print("Not happy text:")
for j in itm.keys():
    print(j, itm[j])

print(sum2)
net = sum1+sum2
print(net)

f.close()
'''