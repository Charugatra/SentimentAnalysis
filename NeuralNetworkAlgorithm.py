import re

from keras.optimizers import Adam
from numpy import array
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import pandas as pd
import numpy as np

def neuralNetwork(text2):
    num_words=6000
    docs=[]
    embedding_size = 8
    df = pd.read_csv('final_data.csv')
    for i in range(len(df)):
        docs.append(re.sub('[^a-zA-Z]', ' ', df.Review[i]))
#print(docs)


    tag=df.Liked

    x_train=docs
    y_train=tag

    x_test=docs
    y_test=tag
    data=x_train+x_test
    tokenizer = Tokenizer(num_words=num_words)

    tokenizer.fit_on_texts(data)

    if num_words is None:
        num_words = len(tokenizer.word_index)
    tokenizer.word_index

    x_train_tokens = tokenizer.texts_to_sequences(x_train)

    x_test_tokens = tokenizer.texts_to_sequences(x_test)


    num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
    num_tokens = np.array(num_tokens)
    np.mean(num_tokens)
    np.max(num_tokens)
    max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
    max_tokens = int(max_tokens)
#print(max_tokens)
    np.sum(num_tokens < max_tokens) / len(num_tokens)
    pad = 'pre'

    x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,
                            padding=pad, truncating=pad)

    x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens,
                           padding=pad, truncating=pad)
    x_train_pad.shape
    x_test_pad.shape

    np.array(x_train_tokens[1])

    idx = tokenizer.word_index
    inverse_map = dict(zip(idx.values(), idx.keys()))


    def tokens_to_string(tokens):
        # Map from tokens back to words.
        words = [inverse_map[token] for token in tokens if token != 0]

        # Concatenate all words.
        text = " ".join(words)

        return text
    model = Sequential()

    embedding_size = 8
    model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))
    model.add(GRU(units=16, return_sequences=True))
    model.add(GRU(units=8, return_sequences=True))
    model.add(GRU(units=4))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    model.fit(x_train_pad, y_train,validation_split=0.05, epochs=3, batch_size=64)
    result = model.evaluate(x_test_pad, y_test)
    print("Accuracy: {0:.2%}".format(result[1]))

    y_pred = model.predict(x=x_test_pad[0:6000])
    #print(y_pred)
    y_pred = y_pred.T[0]
    cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])
    cls_true = np.array(y_test[0:6000])

    layer_embedding = model.get_layer('layer_embedding')
    weights_embedding = layer_embedding.get_weights()[0]
    print(weights_embedding.shape)
    token_good = tokenizer.word_index['good']
    token_great = tokenizer.word_index['great']
    weights_embedding[token_good]
    weights_embedding[token_great]
    token_bad = tokenizer.word_index['bad']
    token_horrible = tokenizer.word_index['horrible']

    incorrect = np.where(cls_pred != cls_true)
    incorrect = incorrect[0]
    #text2=input("enter a string")
    # text2 = "This restaurant is fantastic! I really like it because it is so good!"
    # text1 = "Good restaurant!"
    # text3 = "Maybe I like this restaurant."
    # text4 = "Meh ..."
    # text5 = "If I were a drunk teenager then this restaurant might be good."
    # text6 = "Bad restaurant!"
    # text7 = "Not a good restaurant!"
    # text8 = "This restaurant really sucks! Can I get my money back please?"
    texts = [text2]

    token=[]
    pad='pre'
    tokens = tokenizer.texts_to_sequences(texts)

    tokens_pad = pad_sequences(tokens, maxlen=max_tokens,padding=pad, truncating=pad)
    tokens_pad.shape





    a = model.predict(tokens_pad)
    print(a)
    val=round(a[0][0])
    print(val)


    if val>0:
        print("Neural Network Algorithm : positive")
        tag = "pos"
        return tag
    else:
        print("Neural Network Algorithm : negative")
        tag = "neg"
        return tag