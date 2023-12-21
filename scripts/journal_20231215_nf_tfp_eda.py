# %%
from startup import np, pd, plt, sns

import os
# BACKEND = 'jax'
BACKEND = 'tensorflow'
os.environ['KERAS_BACKEND'] = BACKEND
import tensorflow as tf
if BACKEND == 'jax':
    import tensorflow_probability.substrates.jax as tfp
else:
    import tensorflow_probability as tfp

# import keras_core as keras
from keras_core import ops
from sklearn.datasets import make_moons
from dataclasses import dataclass
# %%
tfd = tfp.distributions
tfb = tfp.bijectors
keras = tf.keras
Dense = keras.layers.Dense
# %%
# Base Distribution: 2D Normal
base_dist = tfd.Normal(loc=[0., 0], scale=1.)

# %%
# Data (destination) distribution: banana shape
data_dist = tfd.JointDistributionNamed(
    dict(
        x_2=tfd.Normal(loc=0., scale=4., name='x_2'),
        x_1=lambda x_2: tfd.Normal(loc=0.25*ops.square(x_2), scale=1., name='x_1')
    ),
    use_vectorized_map=True, batch_ndims=0
)


# %%
def dist_to_df(dist, n_samples):
    samples = dist.sample(n_samples)
    if len(dist.event_shape) > 0:
        # Joint distribution returns a list (...Sequential) or dict (...Named)
        # Can return other structure in general
        # See https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/JointDistribution
        samples = ops.stack(
            [samples['x_1'], samples['x_2']],
            axis=1).numpy()
    else:
        samples = samples.numpy()
    return pd.DataFrame(samples, columns=['x_1', 'x_2'])


# %%
# Example transformation of base distribution
transformed_base = tfd.TransformedDistribution(
    base_dist,
    tfb.Chain(
        # NB: chain evaluated from last element to first
        # i.e. shift then scale here
        [tfb.Scale([10, 2.]), tfb.Shift([1., 0.])]
    )
)

# %%
# Example transformation on each component of data distribution
# Uses JointMap
transformed_dest = tfd.TransformedDistribution(
    data_dist,
    tfb.JointMap(
        {'x_1': tfb.RationalQuadraticSpline(
            bin_widths=[5, 5, 10],
            bin_heights=[2, 6, 12],
            knot_slopes=[1., 20],
            range_min=0.
        ), 'x_2': tfb.Scale(0.25)}
    )
)

# %%
# n_samples = 1000
# df = pd.concat(
#     [dist_to_df(base_dist, n_samples).assign(label='Base'),
#      dist_to_df(data_dist, n_samples).assign(label='Data'),
#      dist_to_df(transformed_base, n_samples).assign(label='T. Base'),
#      dist_to_df(transformed_dest, n_samples).assign(label='T. Data'),
#      ]
# )

# sns.jointplot(
#     df,
#     x='x_1', y='x_2', hue='label', joint_kws=dict(alpha=0.2)
# )
# plt.show()


# %%
class StackedDense(keras.layers.Layer):
    def __init__(self, n_out, n_hidden, dim_hidden, activation, **kwargs):
        super().__init__(**kwargs)
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.activation = activation
        self._layers = []

    def build(self, input_shape):
        self._layers.append(Dense(self.dim_hidden, activation='relu'))
        for i in range(self.n_hidden - 1):
            self._layers.append(Dense(self.dim_hidden, activation='relu'))
        self._layers.append(Dense(self.n_out, activation=self.activation))

    def call(self, inputs, *args, **kwargs):
        # if len(self._layers) == 0:
        #     self.build(inputs)
        x = self._layers[0](inputs)
        for layer in self._layers[1:]:
            x = layer(x)
        return x


# %%
class BinPosition(keras.layers.Layer):
    def __init__(
            self,
            n_bins=32, interval_width=2, range_min=-1, min_bin_width=1e-3,
            dim_bijector_input=None,
            n_hidden=2, dim_hidden=128,
            **kwargs
    ):
        super().__init__(**kwargs)
        # Parameters of output spline
        self._n_bins = n_bins
        self._interval_width = interval_width  # Sum of bin widths.
        self._range_min = range_min  # Position of first knot.
        self._min_bin_width = min_bin_width  # Bin width lower bound.
        # Bijector input dimension - feels like this should come from __call__ though?
        # Try separating into build function called on first __call__
        # (later) ah, no - keras.layers.Layer expects just the data, it's tfp.Bijector that
        # expects the bijector input dimension.  So could use this if we were constructing in
        # first pass of Bijector call but otherwise need to declare in ctor
        self._dim_bijector_input = dim_bijector_input
        # Parameters of neural network
        self._n_hidden = n_hidden
        self._dim_hidden = dim_hidden
        self.net = None
        self.shape = None

    def build(self, input_shape):
        def _bin_positions(x):
            out_shape = ops.concatenate(
                [np.array(list(ops.shape(x)[:-1])), np.array([self._dim_bijector_input, self._n_bins])],
                0
            )
            x = ops.reshape(x, out_shape)
            return ops.softmax(x, axis=-1) * (
                    self._interval_width - self._n_bins * self._min_bin_width
            ) + self._min_bin_width

        self.net = StackedDense(
            n_out=self._dim_bijector_input * self._n_bins,
            n_hidden=self._n_hidden, dim_hidden=self._dim_hidden, activation=_bin_positions, name='w'
        )
        self.net.build(input_shape=input_shape)
        self.shape = self.net.compute_output_shape(input_shape)

    def call(self, inputs, *args, **kwargs):
        # self._dim_bijector_input = dim_bijector_input
        # if self.net is None:
        #     self.build(*args, **kwargs)
        return self.net(inputs, *args, **kwargs)


class KnotSlope(keras.layers.Layer):
    def __init__(
            self,
            n_bins=32, interval_width=2, range_min=-1, min_slope=1e-3,
            dim_bijector_input=None,
            n_hidden=2, dim_hidden=128,
            **kwargs
    ):
        super().__init__(**kwargs)
        # Parameters of output spline
        self._n_bins = n_bins
        self._interval_width = interval_width  # Sum of bin widths.
        self._range_min = range_min  # Position of first knot.
        self._min_slope = min_slope  # Lower bound for slopes at internal knots.
        # Bijector input dimension - feels like this should come from __call__ though?
        # Try separating into build function called on first __call__
        # (later) ah, no - keras.layers.Layer expects just the data, it's tfp.Bijector that
        # expects the bijector input dimension.  So could use this if we were constructing in
        # first pass of Bijector call but otherwise need to declare in ctor
        self._dim_bijector_input = dim_bijector_input
        # Parameters of neural network
        self._n_hidden = n_hidden
        self._dim_hidden = dim_hidden
        self.net = None
        self.shape = None

    def build(self, input_shape):
        def _slopes(x):
            out_shape = ops.concatenate(
                [
                    np.array(list(ops.shape(x)[:-1])),
                    np.array([self._dim_bijector_input, self._n_bins - 1])
                ], 0
            )
            x = ops.reshape(x, out_shape)
            return ops.softplus(x) + self._min_slope

        self.net = StackedDense(
            n_out=self._dim_bijector_input * (self._n_bins - 1),
            n_hidden=self._n_hidden, dim_hidden=self._dim_hidden,  activation=_slopes, name='s'
        )
        self.net.build(input_shape=input_shape)
        self.shape = self.net.compute_output_shape(input_shape)

    def call(self, inputs, *args, **kwargs):
        # self._dim_bijector_input = dim_bijector_input
        # if self.net is None:
        #     self.build(*args, **kwargs)
        return self.net(inputs, *args, **kwargs)


# %%
@dataclass
class SplineParams:
    bin_widths: BinPosition
    bin_heights: BinPosition
    knot_slopes: KnotSlope

# %%
# Learn mapping from base distribution to data dist
# Example from https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/RationalQuadraticSpline
# Modified to make it more compatible with keras Model object

nsplits = 3


xs = np.random.randn(10, 15).astype(np.float32)  # Keras won't Dense(.)(vec).
splines = [
    SplineParams(BinPosition(dim_bijector_input=15-5*i, name=f'bin_width_{i}'),
                 BinPosition(dim_bijector_input=15-5*i, name=f'bin_height_{i}'),
                 KnotSlope(dim_bijector_input=15-5*i, name=f'knot_slope_{i}')
                 ) for i in range(nsplits)
]



# %%
def spline_flow():
    stack = tfb.Identity()
    for i in range(nsplits):
        bin_widths = splines[i].bin_widths
        bin_heights = splines[i].bin_heights
        knot_slopes = splines[i].knot_slopes
        # Try building all SplineParams to avoid  problems with tfp not recognising keras.layers.Layers
        # as tf.Module
        _ = bin_widths(inputs=np.arange(15)[np.newaxis, :])
        _ = bin_heights(inputs=np.arange(15)[np.newaxis, :])
        _ = knot_slopes(inputs=np.arange(15)[np.newaxis, :])

        stack = tfb.RealNVP(
            num_masked=5 * i,
            bijector_fn=tfb.RationalQuadraticSpline(
                bin_widths=bin_widths,
                bin_heights=bin_heights,
                knot_slopes=knot_slopes
            )
        )(stack)
    return stack


stack = spline_flow()

# ys = stack.forward(xs)
# ys_inv = stack.inverse(ys)  # ys_inv ~= xs

# %%
x_m, y_m = make_moons(10_000)


# %%
# Define neural spline flow bijector for moons dataset

# %%
class MoonModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.splines = [
            SplineParams(
                BinPosition(dim_bijector_input=1, name='bw_0'),
                BinPosition(dim_bijector_input=1, name='bh_0'),
                KnotSlope(dim_bijector_input=1, name='ks_0')
            ),
            SplineParams(
                BinPosition(dim_bijector_input=1, name='bw_1'),
                BinPosition(dim_bijector_input=1, name='bh_1'),
                KnotSlope(dim_bijector_input=1, name='ks_1')
            )
        ]

        # Build spline parameter networks
        for spline in self.splines:
            _ = spline.bin_widths(inputs=np.arange(1)[np.newaxis, :])
            _ = spline.bin_heights(inputs=np.arange(1)[np.newaxis, :])
            _ = spline.knot_slopes(inputs=np.arange(1)[np.newaxis, :])

        nsf = tfb.Chain(
            [
                tfb.Permute([1, 0], name='roll_1'),
                tfb.RealNVP(
                    num_masked=1,
                    bijector_fn=tfb.RationalQuadraticSpline(
                        bin_widths=self.splines[1].bin_widths,
                        bin_heights=self.splines[1].bin_heights,
                        knot_slopes=self.splines[1].knot_slopes,
                        name='rqs_1'
                    ),
                    name='real_nvp_1'
                ),
                # tfb.Permute([1, 0], name='roll_0'),
                # tfb.RealNVP(
                #     num_masked=1,
                #     bijector_fn=tfb.RationalQuadraticSpline(
                #         bin_widths=self.splines[0].bin_widths,
                #         bin_heights=self.splines[0].bin_heights,
                #         knot_slopes=self.splines[0].knot_slopes,
                #         name='rqs_0'
                #     ),
                #     name='real_nvp_0'
                # ),
                # tfb.Scale(),
                # tfb.Shift()
            ],
            name='chain'
        )
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=nsf,
            name='flow'
        )

    def call(self, *inputs):
        return self.flow  # .bijector.forward(*inputs)

# %%
moon_model = MoonModel()

# %%
def moon_loss(y_true, y_rv):
    return -ops.mean(moon_model.flow.log_prob(y_true))

# %%
moon_model.compile(
    loss=moon_loss,
    # optimizer=legacy_adam(learning_rate=1e-3),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
)

# %%
moon_model.fit(x_m, x_m)

# %%