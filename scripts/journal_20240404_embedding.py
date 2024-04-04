# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
# %%
from sentence_transformers import SentenceTransformer

# %%
# Downloads and caches about 2GB of model files first time ran
# Model is only 500MB but safetensors, PyTorch, and ONNX versions are downloaded
# (a minute later) huh.  Now only downloads the safetensors version.
# Maybe I previously installed via call to langchain and that downloads all versions?
# Or version update - may need to pin version for work model.
model = SentenceTransformer('intfloat/e5-base-v2')

# %%
texts = [
    'Hello world', 'Hi there', 'Good bye',
    'Make money fast', 'Trade', 'Give me 1M and you won\'t regret it',
    'I love you', 'I hate you',
    'I went to the bank to cash a cheque', 'I went to the bank to catch a fish'
]

# %%
encodings = model.encode(texts)

# %%
ax = sns.heatmap(encodings @ encodings.T, cmap='viridis')
ax.set_title('Cosine Distance')
plt.show()

# %%
ax = sns.heatmap(pairwise_distances(encodings), cmap='viridis')
ax.set_title('L2 Distance')
plt.show()

# %%
# Needs a larger data set
vects = TSNE(n_components=2, perplexity=3).fit_transform(encodings)
df = pd.DataFrame({'label': texts, 'x': vects[:,0], 'y': vects[:,1]})
ax = sns.scatterplot(df, x='x', y='y', hue='label')
ax.legend().set_bbox_to_anchor((1, 1))
plt.tight_layout()
plt.show()