from os.path import join
import numpy as np

from arenets.arekit.contrib.utils.np_utils.embedding import NpzEmbeddingHelper
from arenets.arekit.contrib.utils.np_utils.vocab import VocabRepositoryUtils

vocab = []
vectors = []
shape = None
dir = "_data/wiki"

embedding_source = join(dir, "model.txt")
print("Reading Embedding: {}".format(embedding_source))
with open(embedding_source, "r") as f:
    for i, line in enumerate(f.readlines()):

        args = line.split()

        if i == 0:
            shape = (int(args[0]), int(args[1]))
            continue

        word = args[0]

        assert(word != "")

        vector = [float(i) for i in args[1:]]
        vocab.append(word)
        vectors.append(vector)

vocabulary_target = join(dir, "vocab.txt")
embedding_target = join(dir, "term_embedding")

VocabRepositoryUtils.save(vocab, vocabulary_target)
NpzEmbeddingHelper.save_embedding(np.concatenate(vectors).reshape(shape), embedding_target)