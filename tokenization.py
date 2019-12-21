import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd

import nltk
nltk.download('wordnet')

data = pd.read_csv("moditweet.csv")
print(data.head())

print(data["message"][0])
for i in range(10278):
 print(nltk.word_tokenize(data["message"][i]))


# Tokenize using the white spaces
for i in range(10278):
 print(nltk.tokenize.WhitespaceTokenizer().tokenize(data["message"][i]))

'''
# Tokenize using Punctuations
for i in range(10278):
 print(nltk.tokenize.WordPunctTokenizer().tokenize(data["message"][i]))
'''

for i in range(10278):
 print(nltk.tokenize.TreebankWordTokenizer().tokenize(data["message"][i]))

for i in range(10278):
 words  = nltk.tokenize.WhitespaceTokenizer().tokenize(data["message"][i])
 df = pd.DataFrame()
 df['OriginalWords'] = pd.Series(words)

 #porter's stemmer
 porterStemmedWords = [nltk.stem.PorterStemmer().stem(word) for word in words]
 df['PorterStemmedWords'] = pd.Series(porterStemmedWords)
 #SnowBall stemmer
 snowballStemmedWords = [nltk.stem.SnowballStemmer("english").stem(word) for word in words]
 df['SnowballStemmedWords'] = pd.Series(snowballStemmedWords)
 print(df)

#LEMMATIZATION
for i in range(10278):
 words  = nltk.tokenize.WhitespaceTokenizer().tokenize(data["message"][i])
 df = pd.DataFrame()
 df['OriginalWords'] = pd.Series(words)
 #WordNet Lemmatization
 wordNetLemmatizedWords = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]
 df['WordNetLemmatizer'] = pd.Series(wordNetLemmatizedWords)
 print(df)

