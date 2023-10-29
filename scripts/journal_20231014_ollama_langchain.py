# %%
import jax.numpy as np
import numpy as onp  # Original NumPy
import pandas as pd
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import keras_core as keras

# %%
(x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data() # num_words=vocab_size)

# %%
# See https://keras.io/api/datasets/imdb/
start_char = 1
oov_char = 2
index_from = 3
word_index = keras.datasets.imdb.get_word_index()
inverted_word_index = {(v + index_from): k for k, v in word_index.items()}
inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"


# %%
def sequence_to_sentence(seq):
    return ' '.join(inverted_word_index[x] for x in seq)

# %%
llm = Ollama(
    model="mistral",
    # callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

# %%
# Test that Ollama server is operating
# llm("Tell me about the history of AI")

# %%
res = [
    llm(
        'Does the following review (delimited by ") express a positive sentiment: '
        + f'"{sequence_to_sentence(seq)}"'
        + 'Please respond with a one sentence answer saying why starting with yes or no as appropriate.'
    ) for seq in x_train[:10]
]

# %%
print(list(zip(y_train[:10], res)))