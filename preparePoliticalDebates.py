from visualize import *
import codecs
from preprocess import *
from os import listdir, path, system

if __name__ == "__main__":

    #load corpus from unzipped dir (download from http://mpqa.cs.pitt.edu/corpora/political_debates/)
    corpusPath = sys.argv[1]
    topics = ["guns", "gayRights", "abortion", "creation", "healthcare", "god" ]
    topicFiles = []

    #for each topic:
    for topic in topics:
        p = path.realpath(corpusPath+"/"+topic)
        posts_pos = []
        posts_neg = []
        #prepare training corpus: collect posts for this topic, preprocess and write to file
        for file in listdir(p):
            if file != "README":
                opened = codecs.open(p+"/"+file, "r", "utf8", errors="ignore") #errors due to BOMS
                pos = True
                for line in opened:
                    if line.startswith("#"):
                        if "#stance=stance2" in line:
                            pos = False
                    else:
                        if pos == True:
                            posts_pos.append(line)
                        else:
                            posts_neg.append(line)
                opened.close()
        print("Loaded %d positive posts for topic '%s'" % (len(posts_pos), topic))
        print("Loaded %d negative posts for topic '%s'" % (len(posts_neg), topic))

        print("Preprocessing...")
        preprocessed = preprocess(" ".join(posts_pos))
        outFile = "data/poldeb/"+topic+".pos.txt.tok"
        topicFiles.append(outFile)
        opened = codecs.open(outFile, "w", "utf8")
        opened.write(preprocessed)
        opened.close()
        print("Wrote posts to file %s" % outFile)

        preprocessed = preprocess(" ".join(posts_neg))
        outFile = "data/poldeb/"+topic+".neg.txt.tok"
        topicFiles.append(outFile)
        opened = codecs.open(outFile, "w", "utf8")
        opened.write(preprocessed)
        opened.close()
        print("Wrote posts to file %s" % outFile)