#plot histogram for word counts
#load word counts from file
#e.g.
#   12846 And
#   9760 shall
#   8942 unto
#   8854 I


import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter

if __name__=="__main__":
    wcFile = sys.argv[1]
    opened = codecs.open(wcFile,"r", "utf8")
    wcDict = dict()
    k = 50
    for line in opened.readlines()[:k]:
        count, word = line.split()
        wcDict[word] = int(count)
    opened.close()
    plotName = wcFile+".wc"+str(k)+".plot.png"

    print wcDict

    counts = [v for k,v in sorted(wcDict.iteritems(), key=itemgetter(1), reverse=True)]
    words = [k for k,v in sorted(wcDict.iteritems(), key=itemgetter(1), reverse=True)]

    #Plot histogram using matplotlib bar()
    indexes = np.arange(0,len(words)*4,4)
    width = 4
    plt.bar(indexes, counts, width, color="green")
    plt.xticks(indexes + width * 0.5, words, rotation=90)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(plotName)
