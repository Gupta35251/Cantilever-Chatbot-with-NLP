import pickle
from gensim.models import Word2Vec

# =========================
# SETTINGS
# =========================
EMBEDDING_DIM = 50

# =========================
# LOAD TOKENIZER
# =========================
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================
# BUILD SENTENCES
# =========================
word_index = tokenizer.word_index

sentences = []

for word in word_index.keys():
    sentences.append([word])

# =========================
# TRAIN WORD2VEC
# =========================
w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=EMBEDDING_DIM,
    window=3,
    min_count=1,
    workers=4,
    epochs=20
)

# =========================
# SAVE MODEL
# =========================
w2v_model.save("word2vec.model")

print("=================================")
print("Word2Vec Training Complete")
print("Vocabulary Size:", len(word_index))
print("Embedding Dimension:", EMBEDDING_DIM)
print("=================================")