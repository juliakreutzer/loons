import argparse
import codecs
import cPickle as pkl
import numpy as np
from scipy import spatial
from operator import itemgetter
import matplotlib as ml
ml.use('Agg')
import matplotlib.pyplot as plt
from os import path
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import mpl_cfaces
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
#from igraph import Graph, plot, Plot, configuration
from sklearn.cluster import AffinityPropagation
from sklearn.cross_decomposition import PLSCanonical, PLSRegression, CCA

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


SIMMETRICS = ["cosine", "euclidean", "cityblock"]
NAME = ""

class Similarity():
    @staticmethod
    def cosine(a, b):
        """
        Cosine similarity
        s = a*b / |a|*|b|
        :param a:
        :param b:
        :return: cosine similarity in [0,1]
        """
#        return np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))
        return 1-spatial.distance.cosine(a,b)

    @staticmethod
    def euclidean(a, b):
        return 1-spatial.distance.euclidean(a,b)

    @staticmethod
    def computeSimMatrix(m, metric):
        """
        For all pairs of rows in input matrix, compute similarities
        :param m:
        :return: matrix that contains similarities for all pairs
        """
        number_rows = m.shape[0]
        s = np.zeros((number_rows, number_rows))
        for i in xrange(number_rows):
            for j in xrange(i+1):
                a = m[i]
                b = m[j]
                s[i][j] = metric(a, b)
                s[j][i] = s[i][j] #FIXME assumption: symmetric
        return s


class Embedding():
    def __init__(self, m, d):
        self.m = m
        self.d = d #word2id
        self.rd = {v:k for k,v in self.d.items()} #id2word
        self.vocab_size, self.word_dim = self.getShape()
        self.checkDims()
        self.unkId = self.d["UNK"]

    def getShape(self):
        v,w = self.m.shape
        return v,w

    def checkDims(self):
        if self.vocab_size != len(self.d):
            #raise ValueError("Matrix and dictionary are not aligned (%d vs %d)" % ( self.vocab_size, len(self.d) ))
            pass
    def __str__(self):
        return "Embedding for %d words with dimensionality %d" % (self.vocab_size, self.word_dim)

    def getSimMatrix(self, similarityMetric="cosine"):
        """
        Compose the similarity matrix for all pairs of rows in the given matrix
        :param similarityMetric:
        :return:
        """
        if similarityMetric not in SIMMETRICS:
            raise ValueError("No valid metric specified (%s)" % similarityMetric)
        metric = None
        if similarityMetric == "cosine":
            metric = Similarity.cosine
        if similarityMetric == "euclidean":
            metric = Similarity.euclidean
        simMatrix = Similarity.computeSimMatrix(self.m, metric)
        return simMatrix

    def getMostSimilar(self, word, simMatrix, k=-1):
        """
        Get the k most similar words (and their indices and similarities) for a given word in a sorted list
        :param word:
        :param simMatrix:
        :return:
        """
        i = self.d.get(word, self.unkId)
        simWords = list()
        for j,sim in enumerate(simMatrix[i]): #all sim scores for this word
            simWords.append((j, self.rd.get(j, self.unkId), sim))
        simWords = sorted(simWords, key=itemgetter(2), reverse=True)#sort list by similarity
        wordList = [self.d[s] for (_,s,_) in simWords]
        return simWords[:k], wordList[:k]


def readOptions():
    parser = argparse.ArgumentParser(description="Load parameters for visualization")
    parser.add_argument("embedding", type=str, nargs=2, help="Embedding matrices to visualize")
    parser.add_argument("word2id", type=str, nargs=2, help="Word to index mapping for embeddings")
    parser.add_argument("name", type=str, help="Model identifier (prefix for plots etc.)")
    parser.add_argument("--similarity", default="cosine", type=str, help="Similarity metric for vector similarity"
                                                                         " (cosine, euclidean, dot)")
    parser.add_argument("--vocab", type=str, help="Vocabulary list for potentially interesting words")
    return parser

def loadEmbedding(embeddingFile, word2idFile):
    """
    Load embedding matrix and corresponding word2id mapping from given files
    :param embeddingFile: numpy 2d array
    :param word2idFile: dictionary mapping words to ids
    :return: matrix m, dictionary d
    """
    oef = open(embeddingFile, "r")
    odf = open(word2idFile, "r")
    m = pkl.load(oef)
    d = pkl.load(odf)
    oef.close()
    odf.close()
    return m, d

def mergeEmbeddings(embedding1, embedding2, mode="union", vocab=None):
    """
    Merge two embeddings, i.e. represent them in common vector space,
    new vocabulary: either union or intersection, or limited to vocabulary list
    :param embedding1:
    :param embedding2:
    :param d1:
    :param d2:
    :param mode:
    :param vocab: if specified, only consider embeddings for these words
    :return:
    """
    d1 = embedding1.d
    d2 = embedding2.d
    #print("d1", len(d1), d1)
    #print("d2", len(d2), d2)

    #find mapping of old and new dimensions, new -> old
    nd1 = dict()
    nd2 = dict()

    md = dict()
    unkId = -1

    commonWords = set()
    if vocab is not None and len(vocab)>1:
        commonWords = set(vocab)
        commonWords.add("UNK")
    elif mode == "union": #take union of dictionaries
        commonWords = set(d1.keys()).union(set(d2.keys()))
    elif mode == "intersection": #take intersection of dictionaries
        commonWords = set(d1.keys()).intersection(set(d2.keys()))
    else:
        raise Exception("No valid mode for merging given")
    if len(commonWords) == 0:
        raise Exception("No overlap between dictionaries")

    print("%d common words: %s" % (len(commonWords), commonWords))
    for wid, w in enumerate(commonWords):
        if w == "UNK":
            unkId = wid
        md[w] = wid #enumerate for new mapping
        nd1[wid] = d1.get(w, d1["UNK"]) #fill nd1 and nd2
        nd2[wid] = d2.get(w, d2["UNK"])

    if unkId == -1:
        raise Exception("Models don't contain UNK")

    #permute embedding matrices
    m1 = permuteMatrixByDict(embedding1.m, nd1, embedding1.d["UNK"])
    m2 = permuteMatrixByDict(embedding2.m, nd2, embedding2.d["UNK"])

    #check: get rep for "chocolate" (old and new)
    #print(embedding1.m[embedding1.d["chocolate"]])
    #print(m1[md["chocolate"]])

    nembedding1 = Embedding(m1, md)
    nembedding2 = Embedding(m2, md)
    return nembedding1, nembedding2


def plotMatrix(s, name="simmatrix"):
    """
    Plot a similarity color map
    :param s:
    :return:
    """
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_title(name)
    plt.imshow(s)
    #plt.pcolor(s)
    ax.axis('tight')
    ax.set_aspect('equal')
    plt.colorbar(orientation='vertical')
    fig.savefig("plots/"+NAME+"."+name+".png")

def permuteMatrixByDict(m, d, unkId):
    """
    Permute rows of matrix as specified by mapping (newID -> oldID)
    :param m:
    :param d:
    :return:
    """
    pm = np.zeros(shape=(len(d), m.shape[1]))
    for i in xrange(len(d)): #fill rows of new matrix
        j = d.get(i, unkId) #old row
        pm[i,:] = m[j]
    return pm

def plotWithLabelsAndColors(low_dim_embs, labels, colors, cmap, filename='plots/'+NAME+'.tsne.png', dimRed="tSNE"):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(10,10))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        m = max(1, max(colors))
        cax = plt.scatter(x, y, c=colors[i], cmap=cmap, s=70, vmin=0, vmax=m)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    #plt.colorbar(cax, extend='min')
    print("\tSaved %s cluster plot in '%s'" % (dimRed, filename))
    plt.savefig(filename)

class SimilarityBasedRanking():
    def __init__(self, emb1, emb2, similarityMetric="cosine", filter=None):
        self.emb1 = emb1
        self.emb2 = emb2
        self.s1 = emb1.getSimMatrix(similarityMetric=similarityMetric)
        self.s2 = emb2.getSimMatrix(similarityMetric=similarityMetric)
        self.metric = similarityMetric
        self.filter = filter

    def getAbsoluteDiffRanking(self):
        diffMatrix = np.abs(self.s1-self.s2)
        #plotMatrix(diffMatrix, nameDiff)
        sumRow = np.array([np.sum(diffMatrix[i]) for i in xrange(diffMatrix.shape[0])])
        sumWords = sorted([(self.emb1.rd[i],sumRow[i]) for i in xrange(len(sumRow))], key=itemgetter(1), reverse=True)
        return sumWords

    def getSquaredDiffRanking(self):
        squaredDiffMatrix = np.square(np.subtract(self.s1,self.s2))
        sumSquaredRow = np.array([np.sum(squaredDiffMatrix[i]) for i in xrange(squaredDiffMatrix.shape[0])])
        sumSquaredWords = sorted([(self.emb1.rd[i],sumSquaredRow[i]) for i in xrange(len(sumSquaredRow))],
                                 key=itemgetter(1), reverse=True)
        return sumSquaredWords

    def getCorrelationDiffRanking(self):
        corrRow = np.array([np.correlate(self.s1[i], self.s2[i]) for i in xrange(self.emb1.vocab_size)])
        corrWords = sorted([(self.emb1.rd[i], corrRow[i]) for i in xrange(len(corrRow))], key=itemgetter(1))
        return corrWords

    def getSimGraphs(self, plotGraphs=True):
        """
        Build graphs representing the similarity between words, nodes = words, edge weight = similarity score
        :param plotGraphs:
        :return:
        """
        g1 = Graph.Full(self.s1.shape[0])
        g2 = Graph.Full(self.s2.shape[0])
        g1.es["weight"] = 1.0 #weighted graph
        g2.es["weight"] = 1.0
        g1.es["color"] = "red"
        g2.es["color"] = "blue"
        g1.vs["color"] = "white"
        g2.vs["color"] = "white"
        edge_widths1 = list()
        edge_widths2 = list()
        for i in xrange(self.s1.shape[0]):         #add edges, weighted by similarity
            for j in xrange(self.s1.shape[1]):
                if i == j:
                    continue
                g1[i,j] = s1[i,j]
                g2[i,j] = s2[i,j]
                if self.metric == "cosine":
                    edge_widths1.append(4*s1[i,j]**100)
                    edge_widths2.append(4*s2[i,j]**100)
                else:
                    edge_widths1.append(s1[i,j])
                    edge_widths2.append(s2[i,j])
        g1.vs["label"] = [w for w,i in sorted(self.emb1.d.items(), key=itemgetter(1))]
        g2.vs["label"] = [w for w,i in sorted(self.emb2.d.items(), key=itemgetter(1))]

        #now both together with multiple edges
        g3 = g1.copy()
        g3.es["color"] = "red"
        g3.es["emb"] = 1
        edge_widths3 = list()
        for i in xrange(self.s1.shape[0]):         #add edges, weighted by similarity
            for j in xrange(self.s1.shape[0]):
                if i == j:
                    continue
                g3.add_edge(i,j,emb=2, color="blue")
                if self.metric == "cosine":
                    edge_widths3.append(4*s1[i,j]**100)
                else:
                    edge_widths3.append(s1[i,j])
        g3.vs["label"] = [w for w,i in sorted(self.emb2.d.items(), key=itemgetter(1))]
        u1 = g1.vs.select(label_eq="UNK")
        u2 = g2.vs.select(label_eq="UNK")
        u3 = g3.vs.select(label_eq="UNK")
        g1.delete_vertices(u1)
        g2.delete_vertices(u2)
        g3.delete_vertices(u3)

        #clip edges that have below average weight
        avg1 = np.average(edge_widths1)
        avg2 = np.average(edge_widths2)
        avg3 = np.average(edge_widths3)
        edge_widths1 = [w if w > avg1 else 0 for w in edge_widths1]
        edge_widths2 = [w if w > avg2 else 0 for w in edge_widths2]
        edge_widths3 = [w if w > avg3 else 0 for w in edge_widths3]

        if plotGraphs == True:
            plotname1 = "plots/"+NAME+".simgrapgh1.png"
            plotname2 = "plots/"+NAME+".simgraph2.png"
            plotname3 = "plots/"+NAME+".simgraph1+2.png"
            plot(g1, plotname1, edge_width=edge_widths1, layout="lgl")
            plot(g2, plotname2, edge_width=edge_widths2, layout="lgl")
            plot(g3, plotname3, edge_width=edge_widths3, layout="lgl")
            print("\tSaved similarity graphs in '%s', '%s', '%s'" % (plotname1, plotname2, plotname3))
        return g1, g2, g3

    def getSimMSTs(self, inverse=True, plotGraph=True, root="UNK"):
        rootId1 = self.emb1.d[root]
        rootId2 = self.emb2.d[root]
        if inverse == True:
            d = -1
        else:
            d = 1
        g1 = minimum_spanning_tree(csr_matrix(d*self.s1))
        g2 = minimum_spanning_tree(csr_matrix(d*self.s2))

        a1 = g1.toarray()
        a2 = g2.toarray()

        if plotGraph==True:
            t1 = Graph()
            t2 = Graph()
            t3 = Graph()
            t1.add_vertices(self.emb1.vocab_size)
            t2.add_vertices(self.emb2.vocab_size)
            t3.add_vertices(self.emb1.vocab_size)
            t1.vs["color"] = "white"
            t2.vs["color"] = "white"
            t3.vs["color"] = "white"
            t1.vs["label"] = [w for w,i in sorted(self.emb1.d.items(), key=itemgetter(1))]
            t2.vs["label"] = [w for w,i in sorted(self.emb2.d.items(), key=itemgetter(1))]
            t3.vs["label"] = t1.vs["label"]
            for i in xrange(a1.shape[0]):
                for j in xrange(a1.shape[1]):
                    if a1[i,j] != 0:
                        t1.add_edge(i,j, weight=a1[i,j], color="blue")
                        t3.add_edge(i,j, weight=a1[i,j], color="blue")
            for i in xrange(a2.shape[0]):
                for j in xrange(a2.shape[1]):
                    if a2[i,j] != 0:
                        t2.add_edge(i,j, weight=a2[i,j], color="red")
                        if t3.are_connected(i,j): #edge in both MSTs
                            t3.es[i,j]["color"] = "black"
                        else:
                            t3.add_edge(i,j, weight=a1[i,j], color="red")
            layout1 = t1.layout_reingold_tilford(mode="in", root=rootId1)
            layout2 = t2.layout_reingold_tilford(mode="in", root=rootId2)
            layout3 = t3.layout_reingold_tilford(mode="in", root=rootId1)
            graphs = [Graph.GRG(10, 0.4) for _ in xrange(5)]
            figure = Plot(bbox=(0,0,2000,1000))
            figure.add(t1, layout=layout1, margin=100, bbox=(0,0,1000,1000))
            figure.add(t2, layout=layout2, margin=100, bbox=(1000,0,2000,1000))
            plotname1 = "plots/"+NAME+".mst_trees.png"
            figure.save(plotname1)
            plotname3 = "plots/"+NAME+".merged_mst.png"
            plot(t3, plotname3 , layout=layout3, bbox=(1000,1000), margin=100)
            print("\tSaved MST plots in '%s', '%s'" % (plotname1, plotname3))

        return t1,t2,t3

class ClusterBasedRanking():
    def __init__(self, emb1, emb2):
        self.emb1 = emb1
        self.emb2 = emb2
        self.s1 = self.emb1.getSimMatrix()
        self.s2 = self.emb2.getSimMatrix()
        self.cluster1 = None
        self.cluster2 = None
        self.word2cluster1 = None
        self.word2cluster2 = None
        self.clusterAffinityPropagation()


    def getAffinityPropagationClusters(self):
        return self.cluster1, self.cluster2

    def clusterAffinityPropagation(self):
        """
        Cluster the embeddings with affinity propagation
        :return:
        """
        affin = AffinityPropagation()
        affin.fit(self.emb1.m)
        aflabels1 = affin.labels_
        afclusters1 = dict()
        word2cluster1 = dict()
        for i,l in enumerate(aflabels1):
            points = afclusters1.setdefault(l,list())
            points.append(self.emb1.rd[i])
        for l,c in afclusters1.items():
            for w in c:
                word2cluster1[w] = l
        self.cluster1 = afclusters1
        self.word2cluster1 = word2cluster1
        affin.fit(self.emb2.m)
        aflabels2 = affin.labels_
        afclusters2 = dict()
        word2cluster2 = dict()
        for i,l in enumerate(aflabels2):
            points = afclusters2.setdefault(l,list())
            points.append(self.emb2.rd[i])
        for l,c in afclusters2.items():
            for w in c:
                word2cluster2[w] = l
        self.cluster2 = afclusters2
        self.word2cluster2 = word2cluster2

    def getClusterRanking(self, filter=None):
        """
        Compute clustering similarity score for each word
        """
        intersectionScored = dict()
        for word, wordId in self.emb1.d.iteritems():
            intersect = set(self.cluster1[self.word2cluster1.get(word,self.word2cluster1["UNK"])]).\
                intersection(self.cluster2[self.word2cluster2.get(word,self.word2cluster2["UNK"])])
            union = set(self.cluster1[self.word2cluster1.get(word,self.word2cluster1["UNK"])]).\
                union(self.cluster2[self.word2cluster2.get(word,self.word2cluster2["UNK"])])
            intersectionScored[word] = len(intersect)/float(len(union))
        clusterRanking = sorted([(w,s, self.word2cluster1[w], self.word2cluster2[w])
                                 for w,s in intersectionScored.iteritems()], key=itemgetter(1))
        if filter is not None:
            clusterRanking = [(w, s, c1, c2) for (w, s, c1, c2) in clusterRanking if w in filter]
        return clusterRanking

    def plotClustersTSNE(self):
        """
        Plot clusters in 2dim tSNE space: Not comparable across embeddings
        :return:
        """
        cmap1 = plt.get_cmap('jet', len(self.cluster1))
        cmap1.set_under('gray')
        tsne1 = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs1 = tsne1.fit_transform(self.emb1.m)
        labels1 = [self.emb1.rd[i] for i in xrange(self.emb1.vocab_size)]
        colors1 = [self.word2cluster1[self.emb1.rd[i]] for i in xrange(self.emb1.vocab_size)]
        plotWithLabelsAndColors(low_dim_embs1, labels1, colors=colors1, cmap=cmap1, filename="plots/"+NAME+".tsne1.png", dimRed="tSNE")

        cmap2 = plt.get_cmap('jet', len(self.cluster2))
        if len(self.cluster2) == 1:
            cmap2 = plt.get_cmap('jet', 2)
        cmap2.set_under('gray')
        tsne2 = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs2 = tsne2.fit_transform(self.emb2.m)
        labels2 = [self.emb2.rd[i] for i in xrange(self.emb2.vocab_size)]
        colors2 = [self.word2cluster2[self.emb2.rd[i]] for i in xrange(self.emb2.vocab_size)]
        plotWithLabelsAndColors(low_dim_embs2, labels2, colors=colors2, cmap=cmap2, filename="plots/"+NAME+".tsne2.png", dimRed="tSNE")

    def plotClustersCCA(self, filter=None):
        """
        Plot clusters in 2dim CCA space: Comparable across embeddings
        :return:
        """
        if len(self.cluster1) <= 1:
            cmap1 = plt.get_cmap('jet', 2)
        else:
            cmap1 = plt.get_cmap('jet', len(self.cluster1))
        cmap1.set_under('gray')
        if len(self.cluster2) <= 1:
            cmap2 = plt.get_cmap('jet', 2)
        else:
            cmap2 = plt.get_cmap('jet', len(self.cluster2))
        cmap2.set_under('gray')
        cca = PLSCanonical(n_components=2)
        cca.fit(self.emb1.m, self.emb2.m)
        m1transformed, m2transformed = cca.transform(self.emb1.m, self.emb2.m)
        labels1 = [self.emb1.rd[i] for i in xrange(self.emb1.vocab_size)]
        colors1 = [self.word2cluster1[self.emb1.rd[i]] for i in xrange(self.emb1.vocab_size)]
        labels2 = [self.emb2.rd[i] for i in xrange(self.emb2.vocab_size)]
        colors2 = [self.word2cluster2[self.emb2.rd[i]] for i in xrange(self.emb2.vocab_size)]
        if filter is not None:
            print("\tFiltering samples to plot")
            filteredIds = [self.emb1.d[w] for w in filter] #get ids for words in filter
            m1transformed = m1transformed[filteredIds]
            m2transformed = m2transformed[filteredIds]
            labels1 = [l for l in labels1 if l in filter]
            labels2 = [l for l in labels2 if l in filter]
        elif m1transformed.shape[0] > 100: #sample indices to display, otherwise it's too messy
            filteredIds = np.random.randint(low=0, high=m1.transformed.shape[0]) #sample filteredIds
            m1transformed = m1transformed[filteredIds]
            m2transformed = m2transformed[filteredIds]
            labels1 = [l for l in labels1 if l in filter]
            labels2 = [l for l in labels2 if l in filter]
        plotWithLabelsAndColors(m1transformed, labels1, colors=colors1, cmap=cmap1, filename="plots/"+NAME+".cca1.png", dimRed="CCA")
        plotWithLabelsAndColors(m2transformed, labels2, colors=colors2, cmap=cmap2, filename="plots/"+NAME+".cca2.png", dimRed="CCA")


    def plotClustersHeatMap(self):
        """
        Plot similarity of embeddings in a heatmap with rows ordered by cluster
        :return:
        """
        s1 = self.emb1.getSimMatrix() #ordered by id
        s2 = self.emb2.getSimMatrix()

        #find mapping: new id -> old id
        nd1 = dict()
        nd2 = dict()

        i = 0
        for clid, words in self.cluster1.iteritems():
            for w in words:
                nd1[i] = self.emb1.d[w]
                i += 1

        i = 0
        for clid, words in self.cluster2.iteritems():
            for w in words:
                nd2[i] = self.emb2.d[w]
                i += 1

        pm1 = permuteMatrixByDict(s1, nd1, self.emb1.d["UNK"]) #ordered by cluster
        pm2 = permuteMatrixByDict(s2, nd2, self.emb2.d["UNK"])

        plotMatrix(pm1, name="cluster1matrix")
        plotMatrix(pm2, name="cluster2matrix")

class CCABasedRanking():
    def __init__(self, emb1, emb2, n=200):
        self.emb1 = emb1
        self.emb2 = emb2
        self.n = n

    def getCCARanking(self, filter=None):
        """
        Compare how far apart words are in projection into common space by CCA
        :return:
        """
        cca = PLSCanonical(n_components=self.n)
        cca.fit(self.emb1.m, self.emb2.m)
        m1transformed, m2transformed = cca.transform(self.emb1.m, self.emb2.m)

        #get distances between vectors
        assert self.emb1.vocab_size == self.emb2.vocab_size
        distDict = dict()
        for i in xrange(self.emb1.vocab_size):
            v1 = m1transformed[i]
            v2 = m2transformed[i]
            w = self.emb1.rd[i]
            distDict[w] = 1-Similarity.euclidean(v1,v2)
        ranked = sorted(distDict.iteritems(), key=itemgetter(1), reverse=True)
        if filter is not None:
            ranked = [(w, s) for (w, s) in distDict.iteritems() if w in filter]
        return ranked

def drawFaces(emb1, emb2, wordRanking, n, reduction="cut"):
    """
    Plot Chernoff faces for n most/less interesting words
    From: https://gist.github.com/aflaxman/4043086
    :param n: if negative: less interesting
    :param reduction:
    :return:
    """
    s1 = None
    s2 = None
    if reduction=="cut":
        s1 = emb1.getSimMatrix()[0:,0:18]
        s2 = emb2.getSimMatrix()[0:,0:18]
    elif reduction=="svd":
        s1 = TruncatedSVD(n_components=k).fit_transform(emb1.getSimMatrix())
        s2 = TruncatedSVD(n_components=k).fit_transform(emb2.getSimMatrix())
    elif reduction=="cca": #use orginal embeddings, not similarity matrix for reduction
        cca = PLSCanonical(n_components=18)
        cca.fit(emb1.m, emb2.m)
        s1, s2 = cca.transform(emb1.m, emb2.m)
    interesting = list()
    name = str(n)+"."+reduction
    if n<0: #plot uninteresting words
        n *= -1
        interesting = [wordRanking[::-1][i] for i in xrange(n)]
    else:
        interesting = [wordRanking[i] for i in xrange(n)]
    fig = plt.figure(figsize=(11,11))
    c = 0
    for i in range(n):
        word = interesting[i]
        j = emb1.d[word]
        ax = fig.add_subplot(n,2,c+1,aspect='equal')
        mpl_cfaces.cface(ax, *s1[j]) #nice for similarity matrix *s1[j][:18]
        ax.axis([-1.2,1.2,-1.2,1.2])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(word)
        ax2 = fig.add_subplot(n,2,c+2,aspect='equal')
        mpl_cfaces.cface(ax2, *s2[j])
        ax2.axis([-1.2,1.2,-1.2,1.2])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title(word)
        c += 2
    plotname = "plots/"+NAME+".cface_s1s2_"+name+".png"
    fig.savefig(plotname)
    print("\tSaved Chernoff faces plot in '%s'" % (plotname))

def writeInterestingAndClustersToFiles(wordRanking, cbr, filename, k=5):
    """
    Write list of interesting words and their clusters to file,
    clusters are represented by subset of their words
    :param wordRanking:
    :param cl1:
    :param cl2:
    :param filename:
    :param k: defines how many cluster samples are printed
    :return:
    """
    opened = codecs.open(filename, "w", "utf8")
    cl1, cl2 = cbr.getAffinityPropagationClusters()
    header = "WORD\tCL1\tCL2\n"
    opened.write(header)
    for word in wordRanking:
        writeStr = "%s\t%s\t|\t%s\n" % (word, ", ".join(cl1[cbr.word2cluster1[word]][:k]), ", ".join(cl2[cbr.word2cluster2[word]][:k]))
        opened.write(writeStr)
    opened.close()

if __name__=="__main__":
    parser = readOptions()
    args = parser.parse_args()
    embedding1, embedding2 = args.embedding
    word2id1, word2id2 = args.word2id
    NAME = args.name
    vocabFile = args.vocab
    vocabList = None

    print("Analysis for embeddings %s and %s" % (embedding1, embedding2))
    print("="*100)

    if vocabFile is not None:
        vocabList = []
        print("Analysis limited to vocabulary in %s" % vocabFile)
        opened = codecs.open(vocabFile, "r", "utf8")
        for line in opened: #one word per line
            vocabList.append(line.strip())
        print ("Vocab contains %d words: %s" % (len(vocabList), vocabList))

    full_path = path.realpath(embedding1)
    p, filename1 = path.split(full_path)
    full_path = path.realpath(embedding2)
    p, filename2 = path.split(full_path)
    name1 = filename1+"."+args.similarity
    name2 = filename2+"."+args.similarity

    print("Loading embeddings")
    m1, d1 = loadEmbedding(embedding1, word2id1)
    m2, d2 = loadEmbedding(embedding2, word2id2)
    emb1 = Embedding(m1, d1)
    emb2 = Embedding(m2, d2)

    print("Merging dictionaries")
    memb1, memb2 = mergeEmbeddings(emb1, emb2, mode="intersection")

    ## Similarity-based analysis ##

    print("Computing similarities")
    #build similarity matrices
    s1 = memb1.getSimMatrix(args.similarity)
    s2 = memb2.getSimMatrix(args.similarity)

    #similarity based rankings
    sbr = SimilarityBasedRanking(memb1, memb2, similarityMetric=args.similarity)

    ## Cluster-based analysis ##

    print("Clustering words")
    #cluster-based ranking
    cbr = ClusterBasedRanking(memb1, memb2)
    cl1, cl2 = cbr.getAffinityPropagationClusters()
    clusterRanked = cbr.getClusterRanking(filter=vocabList)
    #mostInteresting = clusterRanked[0][0]
    print("\tCluster-based ranking: %s " % clusterRanked)
    #print("Most interesting word according to clustering comparison: '%s'" % mostInteresting)

    print("Finding most interesting words")

    ## CCA-based analysis ##
    ccabr = CCABasedRanking(memb1, memb2)
    ccaRanked = ccabr.getCCARanking(filter=vocabList)
    print("\tCCA distance-based ranking: %s " % ccaRanked)
    wordRanking = [w for w,dist in ccaRanked]
    mostInteresting = wordRanking[0]
    print("\tMost interesting word according to distance in CCA space: %s (%f)" % (mostInteresting, ccaRanked[0][1]))
    writeInterestingAndClustersToFiles(wordRanking, cbr, "cluster/"+NAME+"interesting+clusters.txt", k=5)

    print("\tEmbedding 1: cluster of '%s': %s" % (mostInteresting, cl1[cbr.word2cluster1[mostInteresting]]))
    print("\tEmbedding 2: cluster of '%s': %s" % (mostInteresting, cl2[cbr.word2cluster2[mostInteresting]]))

    print("Visualizing clusters")
    #plot clusters in 2dim space
    #cbr.plotClustersTSNE()
    cbr.plotClustersCCA(filter=vocabList)


    #heatmap of similarities with rows ordered by clusters
    cbr.plotClustersHeatMap()

    #get k most similar words to most interesting word
    k = 3
    mostSimilar1, wordList1 = memb1.getMostSimilar(mostInteresting, s1, k=k+1)
    mostSimilar2, wordList2 = memb2.getMostSimilar(mostInteresting, s2, k=k+1)
    listing1 = ["%s (%.2f)" % (word, score) for wid, word, score in mostSimilar1[1:]] #leave out word itself (at index 0)
    listing2 = ["%s (%.2f)" % (word, score) for wid, word, score in mostSimilar2[1:]]

    print("\tEmbedding 1: %d most similar words to '%s': %s" % (k, mostInteresting, listing1))
    print("\tEmbedding 2: %d most similar words to '%s': %s" % (k, mostInteresting, listing2))


    ## Visualize most divergent words ##
    print("Visualizing differences in most interesting words")

    #draw Chernoff faces for most and less interesting k words
    drawFaces(memb1, memb2, wordRanking, k)
    drawFaces(memb1, memb2, wordRanking, -k)

    #plot MST for similarities
    sbr.getSimMSTs(root=mostInteresting)
    sbr.getSimGraphs(plotGraphs=True)

    #NOTE: stuff that does NOT work
    #-rank correlation: the same across embeddings
    #-pca: the same
    #-tsne: slow
    #-scaling: doesn't help
    #-graph visualization: painful
    #-silhouette score prefers baseline model
    #-silhouette score does not find interesting words
    #-mutual information does assume same number of items to classify (doesn't work for baselin model)
