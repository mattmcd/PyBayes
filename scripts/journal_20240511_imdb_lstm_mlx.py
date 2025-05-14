# %%
from startup import np, pd, plt, sns
import mlx
import mlx.core as mx
import mlx.nn as nn
import keras as keras

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
df_train = pd.DataFrame(
    {
        'text': [sequence_to_sentence(s) for s in x_train],
        'token': x_train,
        'label': y_train
    }
)


# %%
class LstmAutoencoder(nn.Module):
    def __init__(self, in_dim: int, out_dims: int):
        super().__init__()

        self.encoder = nn.LSTM(in_dim, 128)
        self.decoder = nn.LSTM(in_dim, 128)
        self.classifier = nn.Linear(128, out_dims)

    def __call__(self, x):
        hidden_e, state_e = self.encoder(x)
        hidden_last = hidden_e[:, -1, :]
        decoder_in = mx.tile(hidden_last, [1, hidden_e.shape[1], 1])
        return decoder_in

# %%
obj = LstmAutoencoder(1, 32)

# %%
x = np.array(df_train.token[0])[:64].reshape(1, -1, 1)

# %%
print(obj(mx.array(x)).shape)

# %%
print(mlx.utils.tree_map(lambda p: p.shape, obj.parameters()))
