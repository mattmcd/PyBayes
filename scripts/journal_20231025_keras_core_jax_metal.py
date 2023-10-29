import jax


def wrap_jax_metal(model):

    model.jit_compile = True
    if jax.lib.__version__ < '0.4.14':
        # jit calls in keras-core fail because
        # donate_argnames argument to jit only available from 0.4.14 onwards.
        # Setting the model train_function and test_function avoids the failing jit call
        # METAL backend doesn't use donation so removed entirely rather than replacing with debug_argnums

        def create_step_function(model_step):
            def one_step(state, data):
                data = data[0]
                return model_step(state, data)

            return one_step

        model.train_function = jax.jit(create_step_function(model.train_step))
        model.test_function = jax.jit(create_step_function(model.test_step))
    if jax.lib.xla_bridge.get_backend().platform == 'METAL':
        # Monkey patch to avoid mhlo custom-call failure to Sharding (a nop with Apple Metal anyway)
        # See https://github.com/google/jax/issues/16287
        def nop_sharding(*args, **kwargs):
            # Inconsistency in keras-core calling _enforce_jax_state_sharding
            # Sometimes uses positional calling, other times keyword arguments
            return args if (args is not None and len(args) > 0) else kwargs.values()

        model._enforce_jax_state_sharding = nop_sharding
    return model
