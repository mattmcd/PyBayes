# %%
import sympy as sp

# %%
x, y = sp.symbols('x y')

# %%
def check_commutative(op):
    a, b = sp.symbols('a b')
    ab = op.subs({x: a, y: b})
    ba = op.subs({x: b, y: a})
    return ab == ba

# %%
def check_associative(op):
    a, b, c = sp.symbols('a b c')
    ab = op.subs({x: a, y: b})
    ab_c = op.subs({x: ab, y: c})
    bc = op.subs({x: b, y: c})
    a_bc = op.subs({x: a, y: bc})
    return ab_c == a_bc

# %%
def check_identity(op):
    a, e = sp.symbols('a e', real=True)
    left_eq = sp.Eq(op.subs({x: e, y: a}), a)
    right_eq = sp.Eq(op.subs({x: a, y: e}), a)
    left_identity = sp.solve(left_eq, e)
    right_identity = sp.solve(right_eq, e)
    return left_identity == right_identity
    # return left_eq , right_eq, sp.simplify(left_eq)

# %%
op1 = sp.Abs(x+y) #  x*y/(x+y+1)
print(check_commutative(op1))
print(check_associative(op1))
print(check_identity(op1))