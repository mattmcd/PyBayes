# %%
import itertools

# %%
def fib(n):
    prev = 0
    current = 1
    for x in range(n):
        yield current
        prev, current = current, prev + current


# %%
for x in fib(20):
    print(x, end=' ')

# %%
fib_list = [x for x in fib(20)]


# %%
def fib_lazy():
    # Return lazy list of all Fibonacci numbers
    prev = 0
    current = 1
    while True:
        yield current
        prev, current = current, prev + current


# %%
[x for x in itertools.islice(fib_lazy(), 20)]

# %%
# f = fib_lazy()
# f.__getitem__ = lambda self, *i: itertools.islice(self, *i)

