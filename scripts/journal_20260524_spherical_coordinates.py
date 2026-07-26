# %%
import sympy as sp


# %%
def create_spherical_coordinates(n, symbol):
    """Create n-dimensional normalised vector in n dimensions

    Parameters:
    - n (int): Dimension of the vector
    - symbol (str): Symbol for the vector components

    Returns:
    - list: List of symbolic expressions representing the vector components
    """
    n -= 1

    if n <= 0:
        raise ValueError("Dimension must be positive")

    if not isinstance(symbol, str):
        raise TypeError("Symbol must be a string")

    symbols = sp.symbols(f"{symbol}_1:{n+1}", real=True)
    coords = []
    coords.append(
        sp.prod([sp.sin(s) for s in symbols])
    )
    for i in range(1, n+1):
        coords.append(
            sp.prod([sp.cos(symbols[i-1])] + [sp.sin(s) for s in symbols[i:]])
        )

    return sp.Matrix(coords)

# %%
create_spherical_coordinates(3, 'x')

# %%
# Goal: find the expression for dot product of two normalised vectors of dimension n in spherical coordinates

def normalised_dot_product(n, symbol1='x', symbol2 = 'y'):
    a = create_spherical_coordinates(n, symbol1)
    b = create_spherical_coordinates(n, symbol2)
    return sp.expand_trig(a.dot(b))

# %%
expr = normalised_dot_product(8)
