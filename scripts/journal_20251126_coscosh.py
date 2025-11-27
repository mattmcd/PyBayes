# %%
# From a r/ScientificComputing discussion
# https://www.reddit.com/r/ScientificComputing/comments/1p2x78o/comment/nqub8nu/
# %%
import sympy as sp
import mpmath

# %%
x = sp.symbols('x')
eq = sp.cos(x)*sp.cosh(x) + 1
sols_20 = [sp.nsolve(eq, sp.N(sp.pi/2*(2*k+1)), prec=20, verify=False) for k in range(30)]
# Need precision 50 for residual to not explode
sols_50 = [sp.nsolve(eq, sp.N(sp.pi/2*(2*k+1)), prec=50, verify=False) for k in range(30)]
res = [eq.subs({x: s}) for s in sols_50]
d = [sols_50[i] - sols_20[i] for i in range(30)]
d_pi = [sols_50[i] - sp.N(sp.pi/2*(2*i+1)) for i in range(30)]

# %%
# Linear series approximation
k = sp.symbols('k', is_integer=True, is_real=True, is_positive=True)
eq_lhs = sp.cos(x)*sp.cosh(x)
eq_lhs_approx = sp.cos(x)*sp.exp(x)
sol_linear = sp.solve(sp.series(eq_lhs, x, sp.pi/2*(2*k+1), 2).removeO() + 1, x)[0].simplify()
sol_approx_linear = sp.solve(
    sp.series(eq_lhs_approx, x, sp.pi/2*(2*k+1), 2).removeO() + 1, x
)[0].simplify()

# %%
print(sol_approx_linear.subs({sp.sin(sp.pi*k): 0}).simplify())

# %%
print(sol_linear.subs({sp.sin(sp.pi*k): 0}).simplify())

# %%
f_sol_linear = sp.lambdify(k, sol_linear)
f_sol_approx = sp.lambdify(k, sol_approx_linear)

# %%
print([f_sol_linear(k) for k in range(30)])

# %%
print([f_sol_approx(k) for k in range(30)])

# %%
print([round(sols_50[k], 14) for k in range(30)])

# %%
# Thought: what about dividing through by cosh
eq2 = sp.cos(x) + 1/sp.cosh(x)
sols_20_2 = [sp.nsolve(eq2, sp.N(sp.pi/2*(2*k+1)), prec=50) for k in range(30)]
res2 = [eq.subs({x: s}) for s in sols_20_2]
d2 = [sols_50[i] - sols_20_2[i] for i in range(30)]

# %%
print(d2)