import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest
'''
# Number of Permutations
permutations = 128

# Number of Recommendations to return
num_recommendations = 1

def preprocess(text):
    text = re.sub(r'[^\w\s]','',text)
    tokens = text.lower()
    tokens = tokens.split()
    return tokens

def get_forest(data, perms):
    start_time = time.time()

    minhash = []

    for text in data['text']:
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

    result = database.iloc[idx_array]['message']

    print('It took %s seconds to query forest.' % (time.time() - start_time))

    return result

db = pd.read_csv('moditweet.csv')
db['message'] = db['message'] + ' ' + db['message']
forest = get_forest(db, permutations)

num_recommendations = 5
title = 'Using a neural net to instantiate a deformable model'
result = predict(title, db, permutations, num_recommendations, forest)
print('\n Top Recommendation(s) is(are) \n', result)
'''
'''''
s1 = "i am sad"
s2 = "i am sad"

shingles1 = set([s1[max(0, i-4):i] for i in range(4, len(s1) + 1)])
shingles2 = set([s2[max(0, i-4):i] for i in range(4, len(s2) + 1)])

print(len(shingles1 & shingles2) / len(shingles1 | shingles2))
'''''

import numpy as np
from datasketch import MinHash, MinHashLSH
from nltk import ngrams

def build_minhash(s):
  '''Given a string `s` build and return a minhash for that string'''
  new_minhash = MinHash(num_perm=256)
  # hash each 3-character gram in `s`
  for chargram in ngrams(s, 3):
    new_minhash.update(''.join(chargram).encode('utf8'))
  return new_minhash

array_one = np.array(['dont', 'mistake', 'a', 'bad', 'day', 'with', 'depression', 'everyone', 'ha'])
array_two = np.array(['mistake', 'bad', 'depression','sad'])
#array_t = np.array(['i', 'am', 'not','sad'])

# create a structure that lets us query for similar minhashes
lsh = MinHashLSH(threshold=0.3, num_perm=256)

# loop over the index and value of each member in array two
for idx, i in enumerate(array_two):
  # add the minhash to the lsh index
  lsh.insert(idx, build_minhash(i))

# find the items in array_one with 1+ matches in arr_two
for i in array_one:
  result = lsh.query(build_minhash(i))
  if result:
    matches = ', '.join([array_two[j] for j in result])
    print(' *', i, '--', matches)