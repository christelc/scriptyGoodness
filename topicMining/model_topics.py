from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from stemming.porter2 import stem
import string
import pickle

# https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730

# NMF and LDA are competing topic modeling algorithms. 
from sklearn.decomposition import NMF, LatentDirichletAllocation

def create_and_display_topics(model, feature_names, no_top_words):
    topicList = enumerate(model.components_)

    # each row represents one topic as an array of words
    topicArray = []

    #for topic_idx, topic in enumerate(model.components_):
    for topic_idx, topic in topicList:
        topicWords = []
        # take the index for the top n sorted by values
        # then find the matching features
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            topicWords.append(feature_names[i])
        print ("Topic ", topic_idx, ":")
        print (" ".join(word for word in topicWords))
        #print (" ".join([feature_names[i]
        #                for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topicArray.append(topicWords)
    return topicArray

#match_topics(newLines, tdata, nmf)
# newLines is the original data
# transformedData is a matrix of how each line reflect each topic (non-zero is match)
# nmf contains the topics 
def match_topics(newLines, transformedData, topicArray):
    print ("Matching Topics...")
    print ("Topic Array:")
    print (topicArray)
    i = 0
    for line in newLines:
        # print the line itself followed by the top-scored topic, along with its score
        outline = line + ", "
        # this should select the one with the highest score
        #TODO: is i actually the index or the value
        # this is a for because we might want to list the top n topics
        #print (transformedData[i])
        #print (transformedData[i].argsort())
        #This reads n from teh tail of the list
        n = 1
        for j in transformedData[i].argsort()[::-1][:n]:
            if (j != 0):
                #score
                #print (transformedData[i][j], ", ")
                #topic index
                #print (j, ", ")
                # topics
                #print (topicArray[j], ", ")
                #outline = outline + transformedData[i][j] + ", " + j + ", " + topicArray[j] + ", "
                outline = '{}{}, {}, {},'.format(outline, transformedData[i][j], j, topicArray[j])
            else :
                outline = outline + "NOSCORE, NOIDX, NOMATCH, "
        i = i+1
        print (outline)

def clean_data(lines) :
    cleanLines = []
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
        words = " ".join(stem(word) for word in cleanLine.split(" "))
        cleanLines.append(words)

    return cleanLines



print ("Reading in file...")
with open ('/Users/christelberg/Floreo/textMining/Topline_ABAPrograms.csv') as f:
#with open ('/Users/christelberg/Floreo/textMining/dirtyData.csv') as f:
    lines = f.read().splitlines()

newLines = clean_data(lines)

print (lines)
print (newLines)

no_features = 1000

# NMF is able to use tf-idf
print ("NMF: Vectorizing (creates bag of words matrix)...")
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1,3), max_features=no_features, stop_words='english')
print ("NMF: fit_transform...")
tfidf = tfidf_vectorizer.fit_transform(newLines)
print ("NMF: get feature names...")
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
print ("LDA: count vectorizer...")
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1,3), max_features=no_features, stop_words='english')
print ("LDA: fit_transform...")
tf = tf_vectorizer.fit_transform(newLines)
print ("LDA: get feature names...")
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 50

# Run NMF
print ("NMF: Run...")
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
print ("LDA: Run...")
#lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)


no_top_words = 10
topics = create_and_display_topics(nmf, tfidf_feature_names, no_top_words)
pickle.dump(topics, open ("topics.out", "wb"))
#display_topics(lda, tf_feature_names, no_top_words)

tdata = nmf.transform(tfidf)

pickle.dump(tdata, open ("tdata.out", "wb"))
match_topics(newLines, tdata, topics)

