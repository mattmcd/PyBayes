# %%
from startup import np, pd, plt, sns, os, Path
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif, SelectKBest
import jax.numpy as jnp
import jax
import jax.random as jrand
import keras
import ydf
# %%
def sample_square(n, theta):
    rng = jrand.PRNGKey(0)
    X = jrand.uniform(rng, (2, n))
    rot_mat = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]])
    return (rot_mat @ X).T

# %%
samples = sample_square(10_000, 0.1)
x = samples[:, 0]
y = samples[:, 1]

fig, ax = plt.subplots()
ax.scatter(x, y)
plt.show()

# %%


# %%
def rot_square_mutual_info(n, theta):
    samples = sample_square(n, theta)
    x = samples[:, 0]
    y = samples[:, 1]
    return jnp.array(mutual_info_regression(x[:, jnp.newaxis], y))

# %%
mutual_info_regression(x[:, jnp.newaxis], y)

# %%
theta = jnp.linspace(0, 2 * np.pi/4, 101)
mi = jnp.array([(lambda t: rot_square_mutual_info(10_000, t))(t) for t in theta])

# %%
fig, ax = plt.subplots()
ax.plot(theta, mi)
plt.show()

# %%
def rot_mutual_info(samples, theta):
    rot_mat = jnp.array([[jnp.cos(theta), -jnp.sin(theta)], [jnp.sin(theta), jnp.cos(theta)]]).T
    samples = samples @ rot_mat
    x = samples[:, 0]
    y = samples[:, 1]
    return mutual_info_regression(x[:, jnp.newaxis], y)

# %%
samples = sample_square(10_000, 0.0)
rot_mutual_info(samples, jnp.pi/4)

# %%
# Fails with traced array error?  Is this because mutual_info_regression is using numpy not jnp?
# mi = jax.vmap(rot_mutual_info, in_axes=(None, 0))(samples, theta, )

# %%
def rot_mutual_info_vect(samples, theta):
    return jnp.array([(lambda t: rot_mutual_info(samples, t))(t) for t in theta])
# %%
def plot_mi(samples, theta):
    mi = rot_mutual_info_vect(samples, theta)
    fig, ax = plt.subplots()
    ax.plot(theta, mi)
    plt.show()

# %%
plot_mi(sample_square(10_000, 0.0), theta)

# %%
radius_squared = (samples[:, 0]-0.5)**2 + (samples[:, 1]-0.5)**2
ind_circle = radius_squared <= 0.5**2
circ_samples = samples[ind_circle, :]
plot_mi(circ_samples, theta)

# %%
fig, ax = plt.subplots()
ax.scatter(circ_samples[:, 0], circ_samples[:, 1])
plt.show()

# %%
mi_square = rot_mutual_info_vect(samples, theta)
mi_square_circ_size = rot_mutual_info_vect(samples*jnp.pi/4, theta)
mi_rect_wide = rot_mutual_info_vect(samples @ jnp.diag(jnp.array([1., 0.5]), 0), theta)
mi_rect_tall = rot_mutual_info_vect(samples @ jnp.diag(jnp.array([0.5, 1.]), 0), theta)
mi_circ = rot_mutual_info_vect(circ_samples, theta)
mi_ellipse = rot_mutual_info_vect(circ_samples @ jnp.diag(jnp.array([0.5, 1.]), k=0), theta)
mi_ellipse2 = rot_mutual_info_vect(circ_samples @ jnp.diag(jnp.array([0.25, 1.]), k=0), theta)

# %%
fig, ax = plt.subplots()
ax.plot(theta, mi_square, label='square')
ax.plot(theta, mi_square_circ_size, label='square - eq area')
ax.plot(theta, mi_rect_wide, label='rectangle - wide')
ax.plot(theta, mi_rect_tall, label='rectangle - tall')
ax.plot(theta, mi_circ, label='circle')
ax.plot(theta, mi_ellipse, label='ellipse 1:2')
ax.plot(theta, mi_ellipse2, label='ellipse 1:4')
ax.set_title('Mutual Information')
ax.set_xlabel('Rotation angle')
ax.legend()
plt.show()

# %%
# MI for finding most important features
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


# %%
x_train_flat = x_train.reshape(-1, 28*28)

# %%
k_best = 28
score_func = mutual_info_classif
# Feels like there should be an easier way to set the score function args?
n_neighbors = 3
score_func = lambda *args, **kwargs: mutual_info_classif(*args, **(kwargs | dict(n_neighbors=n_neighbors)))
skb = SelectKBest(score_func, k=k_best)
X_new = skb.fit_transform(x_train_flat[2000:, ...], y_train[2000:])

# %%
fig, ax = plt.subplots()
ax.imshow(skb.scores_.reshape([28,28]), alpha=0.9)
# plt.imshow(
#     (skb.scores_ >= sorted(skb.scores_)[-k_best]).reshape([28, 28]),
#     alpha=0.2,
# )
ax.set_title(f'Mutual Information Scores')
plt.show()


# %%
sns.displot(skb.scores_, kind='ecdf')
plt.show()

# %%
model = ydf.RandomForestLearner(label='y').train({'X': x_train_flat, 'y':y_train})

# %%
evaluation = model.evaluate({'X': x_test.reshape(-1, 28*28), 'y':y_test})

# %%
print(evaluation.characteristics[0].roc_auc)
# %%
print(evaluation.accuracy)

# %%
df_vi = pd.DataFrame(
    model.variable_importances()['INV_MEAN_MIN_DEPTH'], columns=['v', 'name']
).assign(
    ind=lambda df: df.name.str.extract(r'(\d+)').astype(int)
)

# %%
x_vi = np.ones((28*28, ))*df_vi.v.min()
x_vi[df_vi['ind']] = df_vi['v']
plt.imshow(x_vi.reshape([28, 28]))
plt.title('Variable Importances from Random Forest')
plt.show()

# %%
model_mi = ydf.RandomForestLearner(label='y').train({'X': skb.transform(x_train_flat), 'y':y_train})

# %%
evaluation_mi = model_mi.evaluate({'X': skb.transform(x_test.reshape(-1, 28*28)), 'y':y_test})

# %%
print(evaluation_mi.characteristics[0].roc_auc)
# %%
print(evaluation_mi.accuracy)