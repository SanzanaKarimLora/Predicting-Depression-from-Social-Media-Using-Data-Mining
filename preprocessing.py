import re

import pickle as pkl
import pandas as pd
from pandas import DataFrame

import numpy as np

from matplotlib import pyplot as plt


from nltk import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv("E:/4th year-1st semester/Thesis/Dataset/tweets.csv")
nltk_stop_words = list(stopwords.words('english'))
print(data.head())

def preprocess_word(word):
    # Remove punctuation
    word = word.strip('"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    #word = re.sub(r'(-|\')', '', word)
    return word

def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)

def handle_emojis(text):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' smile ', text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' laugh ', text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', ' love ', text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', ' affection ', text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' sad ', text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', ' cry ', text)
    return text


def preprocess_text(text):
    processed_text = []
    # Convert to lower case
    text = text.lower()
    # HTML removed
    # html_process = BS(text, 'html.parser')
    # text = html_process.get_text()

    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)

    # Replaces URLs with the word URL
    text = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' ', text)
    # Replace @handle with the word USER_MENTION
    text = re.sub(r'@[\S]+', ' ', text)
    # Replaces #hashtag with hashtag
    text = re.sub(r'#(\S+)', r' \1 ', text)
    # Remove RT (retext)
    text = re.sub(r'\brt\b', '', text)
    # Replace 2+ dots with space
    text = re.sub(r'\.{2,}', ' ', text)
    # Strip space, " and ' from text
    # text = text.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    text = handle_emojis(text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Striping numbers from the text
    text = re.sub(r'\d+', '', text)

    words = text.split()

    stemmer = SnowballStemmer('english')

    # Removing Stop Words and Stemming the Words
    for word in words:
        word = preprocess_word(word)
        # if word not in nltk_stop_words and len(word)> 2:

        if is_valid_word(word):
            stemmed_words = stemmer.stem(word)
            processed_text.append(word)

    return ' '.join(processed_text)

data['message'] = data['message'].map(lambda x: preprocess_text(x))
print(data.head())
df = DataFrame(data, columns= ['message', 'label'])
process_csv = df.to_csv('preprocess_tweet.csv')
#preprocess = data['message']

#preprocess.to_csv('preprocess_tweet.csv',index=False)



