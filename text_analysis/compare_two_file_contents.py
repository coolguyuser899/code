from __future__ import division

import codecs
import re
import copy
import collections

import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

import matplotlib

#
# nltk.download('stopwords')

# or download everyting in NLTK, 30 minutes needed
# nltk.download('all')

from nltk.corpus import stopwords

with codecs.open("JaneEyre.txt", "r", encoding="utf-8") as f:
    jane_eyre = f.read()
with codecs.open("WutheringHeights.txt", "r", encoding="utf-8") as f:
    wuthering_heights = f.read()

esw = stopwords.words('english')
esw.append("would")

# ^ $ = begin to end, \w+ matches one or more word characters [a-zA-Z0-9]
word_pattern = re.compile("^\w+$")

# create token counter function
def get_text_counter(text):
    tokens = WordPunctTokenizer().tokenize(PorterStemmer().stem(text))
    tokens = list(map(lambda x: x.lower(), tokens))
    tokens = [token for token in tokens if re.match(word_pattern, token) and token not in esw]
    return collections.Counter(tokens), len(tokens)

# calculate absolute frequency
def make_df(counter, size):
    abs_freq = np.array([el[1] for el in counter])
    rel_freq = abs_freq / size
    index = [el[0] for el in counter]
    df = pd.DataFrame(data=np.array([abs_freq, rel_freq]).T, index=index, columns=["Absolute frequency", "Relative frequency"])
    df.index.name = "Most common words"
    return df

je_counter, je_size = get_text_counter(jane_eyre)
make_df(je_counter.most_common(10), je_size)

je_df = make_df(je_counter.most_common(1000), je_size)
je_df.to_csv("JE_1000.csv")

wh_counter, wh_size = get_text_counter(wuthering_heights)
make_df(wh_counter.most_common(10), wh_size)

wh_df = make_df(je_counter.most_common(1000), wh_size)
wh_df.to_csv("WH_1000.csv")

# compare two documents
all_counter = wh_counter + je_counter
all_df = make_df(wh_counter.most_common(1000), 1)
most_common_words = all_df.index.values

# create data frame with word frequency differences
df_data = []
for word in most_common_words:
    je_c = je_counter.get(word, 0) / je_size
    wh_c = wh_counter.get(word, 0) / wh_size
    d = abs(je_c - wh_c)
    df_data.append([je_c, wh_c, d])

dist_df = pd.DataFrame(data=df_data, index=most_common_words, columns=["JE relative frequency",
                       "WH relative frequency", "Relative frequency difference"])
dist_df.index.name = "Most common words"
dist_df.sort_values("Relative frequency difference", ascending=False, inplace=True)
dist_df.head(10)

dist_df.to_csv("bronte.csv")


