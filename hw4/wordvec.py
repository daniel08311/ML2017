import word2vec
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from adjustText import adjust_text

word2vec.word2phrase('p2/all_text.txt', 'p2/all_phrase', verbose=True)
word2vec.word2vec('p2/all_phrase', 'p2/all_text.bin', size=100, verbose=True)
word2vec.word2clusters('p2/all_text.txt', 'p2/all_clusters.txt', 100, verbose=True)

TRAINED = 800
USED = 80

origin_data = 'p2/all_text.txt'
model = 'p2/all_text.bin'

def plot(used_words, x, y, texts):
    color_array = np.arange(used_words)
    plt.figure(figsize=(15,8))
    plt.scatter(x, y, c=color_array,linewidths=0)
    text = []
    for x, y, txt in zip(x, y, texts):
        text.append(plt.text(x, y, txt))
    return text


def preprocess(model):
    idxs = []
    use_tag = ['JJ', 'NNP', 'NN', 'NNS']
    puncts = [ '>' , "'" , '<' , '.' , ':' , ";" , ',' , "?" , "!" , u"’",'"']
    for i, label in enumerate(model.vocab):
        pos = nltk.pos_tag([label])
        if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tag and all(c not in label for c in puncts)):
            if label is not 'Page':
                idxs.append(i)
                
    return list(model.vocab[idxs]), list(model.vectors[idxs])


word2vec.word2vec(origin_data, model, size=250, verbose=False, window=6, alpha=0.05, iter_=5000, min_count = 5)
model = word2vec.load(model)
vocab, vector = preprocess(model)

tsne = TSNE(n_components=2, random_state=0, learning_rate=1200)
np.set_printoptions(suppress=True)
tsne_2d = tsne.fit_transform(vector[:TRAINED])
text = vocab[:USED]
x = tsne_2d[:USED,0]
y = tsne_2d[:USED,1]

texts = plot(USED, x, y, text)
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='b', lw=0.5))
plt.savefig("p2.jpg")
plt.show()
