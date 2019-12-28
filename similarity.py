import np as np
from nltk.tokenize import word_tokenize
'''
data = "Mars is approximately half the diameter of Earth."
print(word_tokenize(data))

from nltk.tokenize import sent_tokenize

data = "Mars is a cold desert world. It is half the size of Earth. "
print(sent_tokenize(data))
'''
import nltk
import gensim
from gensim.sklearn_api import tfidf
from nltk.tokenize import word_tokenize, sent_tokenize
''''
file_docs = []

with open ('token.csv') as f:
    tokens = sent_tokenize(f.read())
    print(tokens)
    for line in tokens:
        file_docs.append(line)

print("Number of documents:",len(file_docs))


gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in file_docs]


dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

tf_idf = gensim.models.TfidfModel(corpus)
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])


# building the index
sims = gensim.similarities.Similarity('venv/',tf_idf[corpus],
                                        num_features=len(dictionary))


file2_docs = []

with open ('sad.csv') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

print("Number of documents:",len(file2_docs))
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf])

import numpy as np

sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
print(sum_of_sims)


percentage_of_similarity = round(float((sum_of_sims / len(file_docs)) * 100))
print(f'Average similarity float: {float(sum_of_sims / len(file_docs))}')
print(f'Average similarity percentage: {float(sum_of_sims / len(file_docs)) * 100}')
print(f'Average similarity rounded percentage: {percentage_of_similarity}')
'''

import np as np
from nltk.tokenize import word_tokenize

import nltk
import gensim
from tfidf import tfidf
from nltk.tokenize import word_tokenize, sent_tokenize

file_docs = []

with open ('token1.csv') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)
        print(file_docs)


print("Number of documents:",len(file_docs))


gen_docs = [[w.lower() for w in word_tokenize(text)]
            for text in file_docs]
print(gen_docs)


dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print(tf_idf)

for doc in tfidf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

'''''
# building the index
sims = gensim.similarities.Similarity('venv/',tf_idf[corpus],
                                        num_features=len(dictionary))


file2_docs = []

with open ('sad.csv') as f:
    tokens = f.read()
    for line in tokens:
        file2_docs.append(line)

print("Number of documents:",len(file2_docs))
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    query_doc_tf_idf = tf_idf[query_doc_bow]
# print(document_number, document_similarity)
print('Comparing Result:', sims[query_doc_tf_idf])

import numpy as np

sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
print(sum_of_sims)


percentage_of_similarity = round(float((sum_of_sims / len(file_docs)) * 100))
print(f'Average similarity float: {float(sum_of_sims / len(file_docs))}')
print(f'Average similarity percentage: {float(sum_of_sims / len(file_docs)) * 100}')
print(f'Average similarity rounded percentage: {percentage_of_similarity}')
'''''





