import net_config as config
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Flatten, Dropout, Reshape
from keras.layers.embeddings import Embedding

prompt_embedding = Sequential()
prompt_embedding.add(Embedding(config.prompt_count+1, config.prompt_embedding_size, input_length = 1))
prompt_embedding.add(Flatten())
prompt_embedding.add(Dense(config.prompt_hidden_layer_size))
prompt_embedding.add(Activation('sigmoid'))
prompt_embedding.add(Dropout(0.5))

d2v = Sequential()
d2v.add(Reshape((config.essay_embedding_size,), input_shape=(config.essay_embedding_size,)))

d2v.add(Dense(config.essay_hidden_layer_size))
d2v.add(Activation('sigmoid'))
d2v.add(Dropout(0.5))

out_score = Sequential()
out_score.add(Merge([prompt_embedding, d2v], mode='concat'))

out_score.add(Dense(config.hidden_layer_size))
out_score.add(Activation('sigmoid'))
out_score.add(Dropout(0.5))

out_score.add(Dense(config.hidden_layer_size))
out_score.add(Activation('sigmoid'))
out_score.add(Dropout(0.5))

out_score.add(Dense((config.score_range[1] - config.score_range[0])+1))
out_score.add(Activation('softmax'))
out_score.compile(loss='categorical_crossentropy', optimizer=config.optimizer)
