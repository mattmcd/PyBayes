# %%
import numpy as np
import jax
import jax.numpy as jnp
from discopy.cat import Ob, Box, Functor, Category
from discopy.matrix import Matrix

# %%
"""
That is the core design philosophy of DisCoPy: 
Functorial Semantics ($F: \text{Syntax} \to \text{Semantics}$).By splitting the problem using DisCoPy's sub-modules, 
you get a clean, scalable architecture:

Syntax (discopy.cat / discopy.monoidal): You define the category $\mathcal{C}$, its objects (Ob), and arrows (Box) abstractly.

Semantics (discopy.matrix / discopy.tensor): You define a Functor that maps syntactic objects to dimensions and syntactic arrows to hom-matrices.

JAX Acceleration: Because DisCoPy's semantic matrices evaluate to JAX arrays, you can jax.vmap and 
jax.jit the Isbell dual operators $(\mathcal{O} \dashv \mathcal{Spec})$ to evaluate all $2^n$ presheaves in parallel 
on GPUs/Apple Silicon.
"""

# %%
# =====================================================================
# 1. SYNTAX LAYER (discopy.cat)
# Define the abstract objects and hom-relations of category C
# =====================================================================
objects = [Ob("6"), Ob("10"), Ob("15")]
n = len(objects)

# Abstract generating box representing the full Hom-structure C(-, -)
hom_box = Box("Hom_C", Ob("C"), Ob("C"))

# %%
# =====================================================================
# 2. SEMANTICS LAYER (discopy.cat.Functor -> discopy.matrix)
# Map abstract syntax to concrete JAX matrices over 2 (Boolean)
# =====================================================================
# Discrete antichain hom-matrix (identity relation)
hom_matrix = Matrix[bool].id(n)

# Define the evaluation Functor: Syntax -> Semantics
F = Functor(
    ob={Ob("C"): n},
    ar={hom_box: hom_matrix},
    cod=Category(int, Matrix)
)

# Evaluate the abstract hom-structure into a JAX boolean matrix
M_jax = jnp.array(F(hom_box).array, dtype=bool).reshape((n, n))

# %%
# =====================================================================
# 3. JAX-ACCELERATED ISBELL DUALITY (jitted over all presheaves)
# =====================================================================
@jax.jit
def isbell_O_batch(P_batch: jax.Array, M: jax.Array) -> jax.Array:
    """
    Computes O(P) for a batch of presheaves in parallel:
    O(P)_j = \\bigwedge_i (~P_i \\lor M_ij)
    """
    # P_batch: (B, N), M: (N, N) -> (B, N, N)
    implication = (~P_batch[:, :, None]) | M[None, :, :]
    return jnp.all(implication, axis=1)

@jax.jit
def isbell_Spec_batch(Q_batch: jax.Array, M: jax.Array) -> jax.Array:
    """
    Computes Spec(Q) for a batch of copresheaves in parallel:
    Spec(Q)_i = \\bigwedge_j (~Q_j \\lor M_ij)
    """
    implication = (~Q_batch[:, None, :]) | M[None, :, :]
    return jnp.all(implication, axis=2)

@jax.jit
def compute_isbell_fixed_points(M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Generates all 2^N presheaves and computes Spec(O(P)) in a single JAX pass.
    """
    # Generate bitmasks for all 2^n presheaves: shape (2^N, N)
    indices = jnp.arange(1 << n)
    bit_shifts = jnp.arange(n)
    P_all = ((indices[:, None] >> bit_shifts[None, :]) & 1).astype(bool)

    # Double dual: P -> Q -> P_closed
    Q_all = isbell_O_batch(P_all, M)
    P_closed_all = isbell_Spec_batch(Q_all, M)

    return P_closed_all, Q_all

# %%
# =====================================================================
# 4. EXECUTION
# =====================================================================
P_closed, Q_dual = compute_isbell_fixed_points(M_jax)

# Deduplicate unique fixed points
unique_pairs = set()
for p, q in zip(jnp.array(P_closed), jnp.array(Q_dual)):
    extent = tuple(obj.name for i, obj in enumerate(objects) if p[i])
    intent = tuple(obj.name for i, obj in enumerate(objects) if q[i])
    unique_pairs.add((extent, intent))

# Output the result
print(f"Evaluated Isbell Completion for Category '{F}'")
print("-" * 55)
for extent, intent in sorted(unique_pairs):
    print(f"Extent (P): {str(extent):<20} | Intent (Q): {intent}")

# %%
"""
Evaluating Lawvere metric categories with DisCoPy and JAX requires mapping 
distance relations into a enriched semiring format ($[0, \infty], \ge, +$), 
extracting the evaluated Hom-matrix, and vectorizing the 
self-dual Mann iteration over batches of presheaves.
"""


# %%
# =====================================================================
# 1. SYNTAX: Define Abstract Lawvere Category
# =====================================================================
# Abstract points and space category
space_ob = Ob("MetricSpace")
dist_box = Box("d_Tree", space_ob, space_ob)

# =====================================================================
# 2. SEMANTICS: DisCoPy Functor -> Lawvere Hom-Matrix
# =====================================================================
# 4-point tree metric distance matrix (Leaves A, B, C, D)
tree_distances = jnp.array([
    [0.0, 2.0, 4.0, 4.0],
    [2.0, 0.0, 4.0, 4.0],
    [4.0, 4.0, 0.0, 2.0],
    [4.0, 4.0, 2.0, 0.0]
], dtype=jnp.float32)

n = tree_distances.shape[0]

# Define functor mapping syntactic box to continuous distance matrix
F_metric = Functor(
    ob={space_ob: space_ob},
    ar={dist_box: Matrix[float](tree_distances.flatten(), n, n)},
    cod=Category(space_ob, Matrix)
)

# Extract JAX array directly from evaluated Functor
D_jax = jnp.array(F_metric(dist_box).array, dtype=jnp.float32).reshape((n, n))

# %%
# =====================================================================
# 3. JAX ACCELERATED ENGINE: Batched Isbell Duals & Mann Iteration
# =====================================================================
@jax.jit
def isbell_O_batch(f_batch: jax.Array, D: jax.Array) -> jax.Array:
    """
    Computes O(f)_y = max_x (D_xy - f_x) across a batch of presheaves (B, N).
    """
    # D: (1, N_x, N_y), f_batch: (B, N_x, 1) -> diffs: (B, N_x, N_y)
    diffs = D[None, :, :] - f_batch[:, :, None]
    return jnp.max(diffs, axis=1)

@jax.jit
def mann_step_batch(f_batch: jax.Array, D: jax.Array) -> jax.Array:
    """Single vectorized self-dual step: 0.5 * (f + O(f))"""
    return 0.5 * (f_batch + isbell_O_batch(f_batch, D))

@jax.jit
def batch_tight_span_solver(f_inits: jax.Array, D: jax.Array, steps: int = 150) -> jax.Array:
    """Runs JIT-compiled parallel Mann iteration for all candidate presheaves."""
    def body_fn(_, f):
        return mann_step_batch(f, D)
    return jax.lax.fori_loop(0, steps, body_fn, f_inits)

# %%
# =====================================================================
# 4. ROBUST MULTI-SCALE INITIAL SAMPLER
# =====================================================================
def generate_robust_presheaves(D: np.ndarray, num_samples: int = 500) -> jax.Array:
    candidates = []

    # 1. Direct row presheaves
    for i in range(n):
        candidates.append(D[i, :])

    # 2. Pairwise midpoints (Essential for internal Steiner nodes!)
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append(0.5 * (D[i, :] + D[j, :]))

    # 3. Sparse Dirichlet face sampling (alpha < 1 forces points onto boundary faces)
    rng = np.random.default_rng(42)
    sparse_weights = rng.dirichlet(np.ones(n) * 0.2, size=num_samples)
    sparse_samples = sparse_weights @ D
    candidates.extend(sparse_samples)

    return jnp.array(candidates, dtype=jnp.float64)


# Generate 1000+ presheaves and compute fixed points in parallel on GPU/Metal
f_inits = generate_robust_presheaves(tree_distances)
tight_span_points = batch_tight_span_solver(f_inits, D_jax)

# Clean and round results
converged_points = jnp.round(jnp.array(tight_span_points), decimals=3)
unique_vertices = jnp.unique(converged_points, axis=0)

# Filter for full-rank 0D extremal vertices
final_vertices = []
for v in unique_vertices:
    tight_mask = jnp.isclose(v[:, None] + v[None, :], tree_distances, atol=1e-3)
    rows = [jnp.eye(n)[i] + jnp.eye(n)[j] for i in range(n) for j in range(i, n) if tight_mask[i, j]]
    if len(rows) > 0 and np.linalg.matrix_rank(rows) == n:
        final_vertices.append(v)

# %%
# print(f"Evaluated Lawvere Metric Space via DisCoPy Functor: '{F_metric}'")
print(f"Processed {len(f_inits)} parallel presheaves in JAX.")
print("-" * 55)
print("0D Extremal Vertices of Tight Span T(X):")
for v in final_vertices:
    print(f"  f = {tuple(v.round(2).tolist())}")