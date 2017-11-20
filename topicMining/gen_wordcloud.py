#!/usr/bin/env python
"""
Generate a wordcloud from a chunk of text
"""

from os import path
from PIL import Image
from stemming.porter2 import stem
import numpy as np
import matplotlib.pyplot as plt
import string

from wordcloud import WordCloud, STOPWORDS

d = path.dirname(__file__)

# clean_data: remove most punctuation and numbers, apply stemming algorithm
def clean_data(text) :
    cleanText1 = ""
    cleanText2 = ""
    # keep a map of the stemmed "word" stem returns with a value of one of the
    # input words. The stemmed word is usually not a real word, and for a 
    # wordcloud we want one of the representative real words
    stemPrintables = {}

    # replace '/', ':', '' with space
    for c in text:
        if c in string.punctuation:
            if c == '\'':
                # let apostrophes in so you dont mess up contractions
                cleanText1 += c
            elif c == '?':
                cleanText1 += ' '
                cleanText1 += c
            else :
                cleanText1 += ' '
        elif c.isalpha() or c.isspace():
            cleanText1 += c

    # apply stemming algorithm
    #words = " ".join(stem(word) for word in cleanLine.split(" "))

    for word in cleanText1.split() :
        stemmed = stem(word)
        if stemmed not in stemPrintables :
            stemPrintables[stemmed] = word
        cleanText2 += " " + stemPrintables[stemmed]

    return cleanText2

# Read the whole text.
text = open(path.join(d, 'Topline_ABAPrograms.csv')).read()
#cleanText = clean_data(text)
cleanText = text

stopwords = set(STOPWORDS)
#stopwords.add("said")

wc = WordCloud(background_color="white", width=1200, height=1000, max_words=2000, 
               stopwords=stopwords)
# generate word cloud
wc.generate(cleanText)

# store to file
wc.to_file(path.join(d, "wordcloud.png"))

# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()



