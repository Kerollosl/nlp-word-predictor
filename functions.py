# Historical experiment helpers. The documented CLI pipeline is canonical.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional


def fit_tokenizer_on_corpus(file, split):
    # Read test
    with open(file) as f:
        data = f.read()
    # Remove digits and punctuation from each sentence
    cleaned_corpus = [''.join(char for char in sentence if char.isalpha() or char.isspace()) for sentence in
                      data.split(split)]

    # Filter out empty sentences (double punctuation, typos, etc.)
    corpus = [sentence for sentence in cleaned_corpus if sentence]

    # Tokenize all words in the corpus based on their frequency of occurrence
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    return corpus, tokenizer


def n_gram_seqs(corpus, tokenizer):
    """
    Generates a list of n-gram sequences

    Args:
        corpus (list of string): lines of texts to generate n-grams for
        tokenizer (object): an instance of the Tokenizer class containing the word-index dictionary

    Returns:
        input_sequences (list of int): the n-gram sequences for each line in the corpus
    """
    input_sequences = []

    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1,len(token_list)):
          n_gram_sequence= token_list[:i+1]
          input_sequences.append(n_gram_sequence)

    return input_sequences


def pad_seqs(input_sequences, maxlen):
    """
    Pads tokenized sequences to the same length

    Args:
        input_sequences (list of int): tokenized sequences to pad
        maxlen (int): maximum length of the token sequences

    Returns:
        padded_sequences (array of int): tokenized sequences padded to the same length
    """
    ### START CODE HERE
    padded_sequences = np.array(pad_sequences(input_sequences, maxlen=maxlen, padding='pre'))

    return padded_sequences
    ### END CODE HERE


def features_and_labels(input_sequences, total_words):
    features = input_sequences[:,:-1]
    labels = input_sequences[:,-1]
    one_hot_labels = to_categorical(labels, num_classes=total_words)

    return features, one_hot_labels


def create_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1)),
    model.add(Bidirectional(LSTM(100))),
    model.add(Dense(total_words, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def plot_metrics(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

