"""
NLTK - https://www.youtube.com/watch?v=3I6M_6YiB2s

word and sentence tokenizer
parts of speech (pos) tagger
extracting entities

"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ne_chunk, pos_tag
#
# text = "hello world this is a simple test.  Mr. Jack and Ms. Jill went up the hill."
# sents = sent_tokenize(text)
# print(sents)
#
# words = word_tokenize(text)
# print(words)
#
# print(nltk.wordpunct_tokenize(text))
# print(nltk.pos_tag(words))
#
# def entities(text):
#     return ne_chunk(
#         pos_tag(
#             word_tokenize(text)
#         )
#     )
#
# tree = entities("When asked about the comments, Obama told the BBC: ""The UK would not be able to negotiate something with the United States faster than the EU"  )
# tree.pprint()
# tree.draw()