from os.path import join
import numpy as np

words = []
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
            print(shape)
            continue

        word = args[0]
        vector = [float(i) for i in args[1:]]
        words.append(word.strip())
        vectors.append(vector)

vocabulary_target = join(dir, "vocab.txt")
print("Saving Vocabulary: {}".format(vocabulary_target))
with open(vocabulary_target, "w") as f:
    for w in words:
        f.write("{}\n".format(w))

embedding_target = join(dir, "term_embedding")
print("Saving embedding: {}".format(embedding_target))
embedding = np.concatenate(vectors).reshape(shape)
np.savez(embedding_target, embedding)
