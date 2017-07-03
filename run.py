#!/usr/bin/python
import warnings
warnings.filterwarnings('ignore')
import gensim.models as w2v
import sklearn.decomposition as dcmp
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hcluster
import re
import nltk
import json
from sklearn.manifold import TSNE


"""
This python module can be used to visualize words based on word embeddings
"""
__author__ = "Sudipta Kar"
#Original author of this idea is Aubry Cholleton

#model_path_google = '/home/ritual/SK/Works/Booxby/word_embeddings_on_reviews/models/min_15_1000_it/wmd_min15_1000.bin'
#model_path_google = '/home/sk/SK/Works/resources/GoogleNews-vectors-negative300.bin.gz'
model_path_google = '/home/sk/SK/Works/NarrativeAnalysis/experiments/classification/processed_data/embeddings/trained_embeddings.bin'

class SemanticMap:
    def __init__(self, model_path):
        #self.model = w2v.Word2Vec.load_word2vec_format(model_path, binary=True, encoding='utf8')
        self.model = w2v.Word2Vec.load(model_path)

    def __split_words(self, input_string):
        return re.findall(r"[\w']+", input_string)

    def __clean_words(self, words):
        clean_words = []
        for w in words:
            clean_words.append(re.sub(r'\W+', '', w.lower()))
        return clean_words

    def __remove_stop_words(self, words):
        return [w for w in words if not w in nltk.corpus.stopwords.words('english')]

    def __get_non_compositional_entity_vector(self, entity):
        return self.model[entity[0]]

    def __get_compositional_entity_vector(self, entity):
        array = np.array(self.model[entity[0]])
        for ind in range (1, len(entity)):
            array = array + np.array(self.model[entity[ind]])
        return array/len(entity)

    def __get_vector(self, term):
        words = self.__remove_stop_words(self.__clean_words(self.__split_words(term)))

        if len(words) < 1:
            print('All the terms have been filtered.')
            raise
        if len(words) == 1:
            try:
                return self.__get_non_compositional_entity_vector(words)
            except:
                print('Out-of-vocabulary entity')
                raise
        elif len(words) < 4:
            try:
                return self.__get_compositional_entity_vector(words)
            except:
                print('Out-of-vocabulary word in compositional entity')
                raise
        else:
            print('Entity is too long.')
            raise

    def __reduce_dimensionality(self, word_vectors):
        data = np.array(word_vectors)
        #pca = dcmp.PCA(n_components=dimension)
        #pca.fit(data)
        #return pca.transform(data)
        
        tsne_model = TSNE(n_components=2, random_state=0, n_iter=1000 )
        return tsne_model.fit_transform(data)


    def cluster_results(self, data, threshold=0.13):
        return hcluster.fclusterdata(data, threshold, criterion="distance")

    def map_words(self, words, sizes):
        final_words = []
        final_sizes = []
        vectors = []

        for word in words:
            try:
                vect = self.__get_vector(word)
                vectors.append(vect)
                if sizes is not None:
                    final_sizes.append(sizes[words.index(word)])
                final_words.append(word)
            except Exception:
                print('not valid ' + word)

        return vectors, final_words, final_sizes

    def plot(self, vectors, lemmas, clusters, sizes=80):
        if sizes == []:
            sizes = 80
        plt.scatter(vectors[:, 0], vectors[:, 1], s=sizes, c=clusters)
        for label, x, y in zip(lemmas, vectors[:, 0], vectors[:, 1]):
            plt.annotate(
                label,
                xy = (x, y), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

        plt.show()

    def map_cluster_plot(self, words, sizes, threshold):
        vectors, words, sizes = self.map_words(words, sizes)
        vectors = self.__reduce_dimensionality(vectors)
        clusters = self.cluster_results(vectors, threshold)
        self.plot(vectors, words, clusters, sizes)

    def print_results(self, words, clusters):
        print(words)
        print(clusters.tolist())


def save_config(config):
    with open('config.json', 'w') as f:
        json.dump(config, f)
        f.close()

def take_parameters():
    config = {}
    with open('config.json', 'r') as f:
        config = json.load(f)
        f.close()

    print(open('welcome_message.txt', 'r').read().strip().format(
                                    config['vector_path'], config['dr_method'], config['loading_system']))

    while True:
        line = raw_input('> ').strip()

        if line.startswith('-emb'):
            tokens = line.split(' ')
            if len(tokens) !=2:
                print('Invalid command.\n')
            else:
                config['vector_path'] = tokens[1]
                save_config(config)
        elif line == 'EXIT' or line == 'exit':
            exit()

        elif line.startswith('-plot'):
            tokens = line.split()
            words   = []
            if tokens[1] == '-f':
                words = open(tokens[2], 'r').read().split('\n')[:-1]
            else:
                words = tokens[1].split(',')

            plot(words)


if __name__ == "__main__":
    take_parameters()
    cli()
    #mapper = SemanticMap(model_path_google)
    #cli(mapper)
