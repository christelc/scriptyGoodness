from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from stemming.porter2 import stem
import matplotlib.pyplot as plt
import string
import pickle
from os import path

from PIL import Image
from wordcloud import WordCloud, STOPWORDS

d = path.dirname(__file__)

# These words on their own aren't helpful, but combined may be.
# this just excludes them when alone
stoplist = [
        "item",
        "level",
        "play",
        "tact",
        "motor",
        "question"
    ]

# clean_data: remove most punctuation and numbers, apply stemming algorithm
def clean_data(lines) :
    cleanLines = []

    # keep a map of the stemmed "word" stem returns with a value of one of the
    # input words. The stemmed word is usually not a real word, and for a
    # wordcloud we want one of the representative real words
    stemPrintables = {}

    for line in lines:
        # replace '/', ':', '' with space
        cleanLine = ""
        for c in line:
            if c in string.punctuation:
                if c == '\'':
                    # let apostrophes in so you dont mess up contractions
                    cleanLine += c
                elif c == '?':
                    cleanLine += ' '
                    cleanLine += c
                else :
                    cleanLine += ' '
            elif c.isalpha() or c.isspace():
                cleanLine += c
        # apply stemming algorithm
        cleanWords = ""
        for word in cleanLine.split():
            stemmed = stem(word)
            #print ("Stemming: ", word, " ", stemmed)
            if stemmed not in stemPrintables :
                stemPrintables[stemmed] = word
            cleanWords += " " + stemPrintables[stemmed]

        cleanLines.append(cleanWords)

    return cleanLines



# NMF and LDA are competing topic modeling algorithms. Unsupervised.
from sklearn.decomposition import NMF, LatentDirichletAllocation


print ("Reading in file...")
with open ('Topline_ABAPrograms.csv') as f:
    lines = f.read().splitlines()

newLines = clean_data(lines)


#print ("orig:", lines)
#print ("clean:", newLines)

# number of features to extract and use in analysis
no_features = 1000

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(2,4), max_features=no_features, stop_words='english')
matrix = tf_vectorizer.fit_transform(newLines)
tf_feature_names = tf_vectorizer.get_feature_names()

#print (tf_vectorizer.vocabulary_)

freqs = [(word, matrix.getcol(idx).sum()) for word, idx in tf_vectorizer.vocabulary_.items()]
#sort from largest to smallest
print (sorted (freqs, key = lambda x: -x[1]))

# make a map of phrase to count as input to wordcloud
wordToCount = {}
for word, idx in tf_vectorizer.vocabulary_.items():
    wordToCount[word] = matrix.getcol(idx).sum()
#print (wordToCount)

# remove stoplist words (not as part of other phrases, just as-is)
#for word in stoplist:
#    wordToCount.pop(word, None)

# generate a wordcloud from CountVectorizer's word counts
wc = WordCloud(background_color="white", width=1200, height=1000, max_words=2000) 
wc.generate_from_frequencies(wordToCount)

# store to file
wc.to_file(path.join(d, "wordcloud.png"))


# show
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()



