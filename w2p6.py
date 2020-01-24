import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from pandas import DataFrame

import nltk
nltk.download('wordnet')

data = pd.read_csv("moditweet.csv")
#print(data.head())

print(data["message"][8916])
print(nltk.word_tokenize(data["message"][8916]))


# Tokenize using the white spaces

words  = nltk.tokenize.WhitespaceTokenizer().tokenize(data["message"][8916])
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

words = nltk.tokenize.WhitespaceTokenizer().tokenize(data["message"][8916])
df = pd.DataFrame()
df['OriginalWords'] = pd.Series(words)
 #WordNet Lemmatization
wordNetLemmatizedWords = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]
df['WordNetLemmatizer'] = pd.Series(wordNetLemmatizedWords)

print(df)


