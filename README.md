# loons
Training and visualizing contrastive word embeddings:
Given a number of text collections, the goal is to identify those words which are divergent across collections. This is realized by training word embeddings on the collections separately and then comparing them.

Python modules needed for running training and visualization:
- tensorflow
- numpy
- scipy
- matplotlib
- igraph


## Leave-One-Out Negative Sampling (LOONS) ##
This is a modified version of the word2vec Skip-Gram model and its [tensorflow implementation](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/models/embedding/word2vec_optimized.py). 
The negative samples are not drawn from the unigram distribution of the target text collection, but from all other collections ("leave-one-out").

To train a LOONS model, use the following call to the loons.py script (for more parameters see the code):
`python loons.py --train_data <data/dir/text_A> --save_path <model/> --name <MyModel>` 
The embeddings are trained on the text `text_A` whereas negative samples are drawn from all other .txt documents in the same directory, say `data/dir` contains three texts `text_A`, `text_B` and `text_C`, then training on `text_A` will involve sampling from `text_B` and `text_C`.
The trained word embeddings are stored in `embeddings/` (pickled) and the according word dictionaries (mapping words to indices in the embedding matrices) are stored in `dicts/`.  

## Visualization ##
`visualize.py` allows to identify words which are divergent across embeddings by a number of techniques. An example is provided in the main method of the script. The basic proceeding is the following:

1. The vocabularies of both embeddings are merged (either by union, intersection, or filtering to given vocabulary)
2. The rows of the embeddings are accordingly selected and permuted, such that corresponding rows represent the same words.
3. Both embeddings are clustered (e.g. with scikit-learn's Affinity Propagation)
4. Both embeddings are projected into the same vector space via CCA.
5. In this space, distance metrics allow to directly compare the position, i.e. the meaning, of words across embeddings.
6. For a selection of the most divergent words, the cluster members are listed and a number of visualizations are plotted: Maximum Spanning Trees over similarity matrices, Chernoff faces, and scatter plots in the CCA-projected space

For a full run of this analysis, execute

`python visualize <embedding1.pkl> <embedding2.pkl> <w2i1.pkl> <w2i2.pkl> --name <MyAnalysis>`

If you specify the parameter `--vocab <MyList.txt>` this list will provide a pre-selection on potentially interesting words.

Any questions? Ideas or even use-cases? --> kreutzer@cl.uni-heidelberg.de
