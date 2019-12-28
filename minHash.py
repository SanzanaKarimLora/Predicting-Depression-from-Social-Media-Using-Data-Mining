import numpy as np
import pandas as pd
import re
import time
#from datasketch import MinHash, MinHashLSHForest
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
'''
s1 = ["sad","i","am"]
s2 = ["sad"]

shingles1 = set([s1[max(0, i-4):i] for i in range(4, len(s1) + 1)])
shingles2 = set([s2[max(0, i-4):i] for i in range(4, len(s2) + 1)])
try:
 print(len(shingles1 & shingles2) / len(shingles1 | shingles2))
except ZeroDivisionError:
    z = 0
'''

'''''
import numpy as np
from datasketch import MinHash, MinHashLSH
from nltk import ngrams

def build_minhash(s):
 
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

''''''
from datasketch import MinHash

data1 = ['dont', 'mistake', 'a', 'bad', 'day', 'with', 'depression', 'everyone', 'ha']
data2 = ['just', 'had', 'a', 'real', 'good', 'moment', 'i', 'miss', 'him', 'so', 'much']
data3 = ['i', 'am', 'really', 'unhappy']
sad = ['mistake', 'bad', 'depression','sad', 'miss','unhappy']
happy = ['happy','well','good','excellent','enjoy','wow', 'wonder', 'miracle','great','fine','satisfy','favor']

m1, m2 = MinHash(), MinHash()

for d in data1:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))

for d in data2:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))

s1 = set(data1)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data2)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

# This will give better accuracy than the default setting (128).
m = MinHash(num_perm=256)

# The makes m1 the union of m2 and the original m1.
print(m1.merge(m2))

# Returns the estimation of the cardinality of
# all data values seen so far.
print(m.count())
'''''

'''
from datasketch import MinHash
import numpy as np
import pandas as pd
import re
import time
from datasketch import MinHash, MinHashLSHForest



data1 = ['dont', 'mistake', 'a', 'bad', 'day', 'with', 'depression', 'everyone', 'ha']
data2 = ['just', 'had', 'a', 'real', 'good', 'moment', 'i', 'miss', 'him', 'so', 'much']
data3 = ['i', 'am', 'really', 'unhappy']
sad = ['mistake', 'bad', 'depression','sad', 'miss','unhappy']
happy = ['happy','well','good','excellent','enjoy','wow', 'wonder', 'miracle','great','fine','satisfy','favor']

m1, m2 = MinHash(), MinHash()


s1 = set(data1)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

minhash = []

# This will give better accuracy than the default setting (128).
m = MinHash(num_perm=256)

# The makes m1 the union of m2 and the original m1.
print(m1.merge(m2))

# Returns the estimation of the cardinality of
# all data values seen so far.
print(m.count())
for d in data2:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for d1 and d2 is", m1.jaccard(m2))
import csv
with open("token_tweet.csv", "r") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        tweets = row['message']
        for d in tweets:
            m1.update(d.encode('utf8'))
        for d in sad:
             m2.update(d.encode('utf8'))
        print("Estimated Jaccard for data1 and data2 is", m1.jaccard(m2))

        print(tweets)
'''

from datasketch import MinHash

data1 = ['dont', 'mistake', 'a', 'bad', 'day', 'with', 'depression', 'everyone', 'ha']
data2 = ['just', 'had', 'a', 'real', 'good', 'moment', 'i', 'miss', 'him', 'so', 'much']
data3 = ['i', 'am', 'really', 'unhappy']
data4 = ['depression']
data5 = ['night', 'sweet', 'dream', 'to', 'you']
data6 = ['depression', 'and', 'sadness']
data7 = ['depression', 'really', 'a', 'mf']
data8 = ['the', 'only', 'thing', 'that','there','for','me','is','my','depression']
data9 = ['and','depression']
data10 = ['thats', 'a', 'good', 'thing']
sad = ['mistake', 'bad', 'depression','sad', 'miss','unhappy']
happy = ['happy','well','good','excellent','enjoy','wow', 'wonder', 'miracle','great','fine','satisfy','favor','sweet']

m1, m2 = MinHash(), MinHash()

for d in data1:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data1 and sad is", m1.jaccard(m2))

for d in data2:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data2 and sad is", m1.jaccard(m2))

for d in data3:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data3 and sad is", m1.jaccard(m2))

for d in data4:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data4 and sad is", m1.jaccard(m2))

for d in data5:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data5 and sad is", m1.jaccard(m2))

for d in data6:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data1 and sad is", m1.jaccard(m2))

for d in data7:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data2 and sad is", m1.jaccard(m2))

for d in data8:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data3 and sad is", m1.jaccard(m2))

for d in data10:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data4 and sad is", m1.jaccard(m2))

for d in data9:
    m1.update(d.encode('utf8'))
for d in sad:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data9 and sad is", m1.jaccard(m2))


s1 = set(data1)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data2)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data3)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)


s1 = set(data4)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)


s1 = set(data5)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data6)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data7)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data8)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)


s1 = set(data10)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)



s1 = set(data9)
s2 = set(sad)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data9 and sad is", actual_jaccard)


for d in data1:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data1 and sad is", m1.jaccard(m2))

for d in data2:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data2 and sad is", m1.jaccard(m2))

for d in data3:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data3 and sad is", m1.jaccard(m2))

for d in data4:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data4 and sad is", m1.jaccard(m2))

for d in data5:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data5 and sad is", m1.jaccard(m2))

for d in data6:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data1 and sad is", m1.jaccard(m2))

for d in data7:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data2 and sad is", m1.jaccard(m2))

for d in data8:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data3 and sad is", m1.jaccard(m2))

for d in data10:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data4 and sad is", m1.jaccard(m2))

for d in data9:
    m1.update(d.encode('utf8'))
for d in happy:
    m2.update(d.encode('utf8'))
print("Estimated Jaccard for data9 and sad is", m1.jaccard(m2))


s1 = set(data1)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data2)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data3)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)


s1 = set(data4)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)


s1 = set(data5)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data6)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data7)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)

s1 = set(data8)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)


s1 = set(data10)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data1 and sad is", actual_jaccard)



s1 = set(data9)
s2 = set(happy)
actual_jaccard = float(len(s1.intersection(s2)))/float(len(s1.union(s2)))
print("Actual Jaccard for data9 and sad is", actual_jaccard)

'''
###run properly###
from datasketch import MinHash
m1, m2 = MinHash(), MinHash()
with open ('token.csv') as f:
    tokens = f.read()

    for line in tokens:
        #file_docs.append(line)
        m1.update(line.encode('utf8'))

with open('sad.csv') as s:
    sad = s.read();
    for d in sad:
        m2.update(d.encode('utf8'))

print("Estimated Jaccard for dataset and sad text is", m1.jaccard(m2))
'''



