import requests
from bs4 import BeautifulSoup

########################################################
#     Get all links of old posts from blog spot
#     using the requests and  BeautifulSoup module
########################################################
def getAllPostLinks(URL, listOfLinks):    
    response = requests.get(URL)    
    soup = BeautifulSoup(response.content,"html.parser")
    for a in soup.find_all('a'):
        try:
            URL = a['href']
            title = a['title']
            if title  == "Older Posts":
                listOfLinks.append(URL)
                getAllPostLinks(URL,listOfLinks)
        except:       
            title = ""

URL = "http://doxydonkey.blogspot.in/"
listOfLinks = []
getAllPostLinks(URL,listOfLinks)

########################################################
#     Get all the articles from all links which we got 
#     from old posts from blog spot
#     using the requests and  BeautifulSoup module
########################################################

def getArticleText(testURL):
    response = requests.get(testURL)    
    soup = BeautifulSoup(response.content,"html.parser")
    myDivs = soup.find_all("div",{"class":"post-body"})

    posts = []
    for div in myDivs:                
        posts += map(lambda p: p.text.encode('ascii',errors='replace').decode('utf-8').replace("?"," "), div.find_all('li'))
    return posts

articles = []
for link in listOfLinks:
    articles+=getArticleText(link)

print(len(articles))

########################################################
#     
#       Cluster using machine learning algorithm (k-means)
#     
########################################################

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(articles)

km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
km.fit(X)

np.unique(km.labels_, return_counts=True)

text = {}
for i, cluster in enumerate(km.labels_):
    oneDoc = articles[i]
    if cluster not in text.keys():
        text[cluster] = oneDoc
    else:
        text[cluster] += oneDoc
########################################################
#     
#      Find the themes of each clusters
#     
########################################################
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import defaultdict
from string import punctuation
from heapq import nlargest
import nltk

_stopwords = set(stopwords.words('english') + list(punctuation)+["million","billion","year","millions","billions","y/y","'s","''","``"])

keywords = {}
counts = {}

for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent) 
    keywords[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster] = freq  

# get the top and unique keywords from each clusters
unique_keys = {}
for cluster in range(3):
    other_clusters = list(set(range(3)) - set([cluster]))
    keys_other_clusters = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique = set(keywords[cluster]) - keys_other_clusters
    unique_keys[cluster] = nlargest(10, unique, key=counts[cluster].get)

########################################################
#     
#      Classify a new Article
#     
########################################################

##newArticle = "Not Anymore. When Uber picked this former Rust Belt town as the inaugural city for its driverless car experiment, Pittsburgh played the consummate host. “You can either put up red tape or roll out the red carpet,” Bill Peduto, the mayor of Pittsburgh, said in September. “If you want to be a 21st-century laboratory for technology, you put out the carpet.” Nine months later, Pittsburgh residents and officials say Uber has not lived up to its end of the bargain. Among Uber’s perceived transgressions: The company began charging for driverless rides that were initially pitched as free. It also withdrew support from Pittsburgh’s application for a $50 million federal grant to revamp transportation. And it has not created the jobs it proposed in a struggling neighborhood that houses its autonomous car testing track. The deteriorating relationship between Pittsburgh and Uber offers a cautionary tale, especially as other cities consider rolling out driverless car trials from Uber, Alphabet’s Waymo and others"
newArticle = ""
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=10)
classifier.fit(X, km.labels_)

test_vec = vectorizer.transform([newArticle.encode('ascii',errors='ignore').decode()])

classifier.predict(test_vec)



