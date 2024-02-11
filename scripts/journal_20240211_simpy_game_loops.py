# %%
from startup import np, pd, plt, sns
import simpy
# %%
from mattmcd.simpy import Source, Sink, Converter, MultiConverter

# %%
env = simpy.Environment()
source = Source(env)
sink = Converter(env, source)
env.run(until=15)

# %%
env = simpy.Environment()
source_1 = Source(env, name='A')
source_2 = Source(env, name='B')
sink = MultiConverter(env, (source_1, source_2), (6, 2))
env.run(until=15)
