from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'weights/embeddings/glove.840B.300d.txt'
word2vec_output_file = 'weights/embeddings/glove.840B.300d.vec'
glove2word2vec(glove_input_file, word2vec_output_file)
