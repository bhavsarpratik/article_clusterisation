from bs4 import BeautifulSoup
import nltk
from heapq import nlargest
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from heapq import nlargest

customStopWords = set(stopwords.words('english') + list(punctuation))

blogUrl = "http://doxydonkey.blogspot.in/"


def getAllDoxyDonkeyPosts(url, links):
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data, 'html.parser')
    for a in soup.findAll('a'):
        try:
            url = a['href']
            title = a['title']
            if title == "Older Posts":
                links.append(url)
                print(title)
                getAllDoxyDonkeyPosts(url, links)
        except:
            title = ""
    return

links = []

getAllDoxyDonkeyPosts(blogUrl, links)

print('Number of articles downloaded: %d' % len(links))


def getDoxyDonkeyText(url):
    http = urllib3.PoolManager()
    response = http.request('GET', url)
    soup = BeautifulSoup(response.data, 'html.parser')
    mydivs = soup.findAll("div", {"class": 'post-body'})

    posts = []

    for div in mydivs:
        print('Downloading..')
        posts += map(lambda p: p.get_text().replace('\xa0', ' '),
                     div.findAll("li"))
    return posts

doxyDonkeyPosts = []

for link in links:
    doxyDonkeyPosts += getDoxyDonkeyText(link)

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
# min_df & max_df : When using a float in the range [0.0, 1.0] they refer to the document frequency.
# That is the percentage of documents that contain the term. When using an int
# it refers to absolute number of documents that hold this term.

X = vectorizer.fit_transform(doxyDonkeyPosts)

ncluster = 5  # number of desired clusters
km = KMeans(n_clusters=ncluster, init='k-means++',
            max_iter=200, n_init=5, verbose=True)
# n_init: Number of time the k-means algorithm will be run with different centroid
# seeds. The final results will be the best output of n_init consecutive runs
# in terms of inertia.

km.fit(X) #vectorizis and assigns cluster

# print(np.unique(km.labels_, return_counts=True))

# creating dictionary text of article and its cluster
text = {}
for i, cluster in enumerate(km.labels_):
    oneDocument = doxyDonkeyPosts[i]
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument

keywords = {}
counts = {}

# finding characteristic of the cluster
for cluster in range(ncluster):
    words = word_tokenize(text[cluster].lower())
    filteredWords = [w for w in words if w not in customStopWords]
    freq = FreqDist(filteredWords)
    keywords[cluster] = nlargest(100, freq, key=freq.get)
    counts[cluster] = freq

unique_keys = {}
for cluster in range(ncluster):
    other_clusters = list(set(range(3)) - set([cluster]))
    keys_other_clusters = set(keywords[other_clusters[0]]).union(
        set(keywords[other_clusters[1]]))
    unique = set(keywords[cluster]) - keys_other_clusters
    unique_keys[cluster] = nlargest(10, unique, key=counts[cluster].get)

print(unique_keys)
