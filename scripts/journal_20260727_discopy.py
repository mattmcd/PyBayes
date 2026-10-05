# %%
"""
DisCoPy is a library for working with diagrams in Python.
"""

# %%
from discopy.cat import Ob, Box, Functor
from discopy.monoidal import Ty, Diagram
# %%
x, y, z = [Ob(s) for s in 'xyz']
f, g, h = [
    Box(m, dom=d, cod=r) for m,d,r in [('f', x, y), ('g', y, z), ('h', z, x)]
]

# %%
F = Functor(ob={x: y, y: z, z: x}, ar={f: g, g: h})
assert F(f >> g) == F(f) >> F(g) == g >> h

# %%
import numpy as np
from discopy.cat import Ob
from discopy.matrix import Matrix


class EnrichedCategory:
    """
    Represents a finite 2-enriched category (poset/relation) in DisCoPy.
    The hom-set relation C(x, y) is encoded as a discopy.matrix.Matrix[bool].
    """

    def __init__(self, names: list[str], hom_matrix: Matrix):
        self.objects = [Ob(name) for name in names]
        self.n = len(names)
        self.hom = hom_matrix

    def _get_matrix(self) -> np.ndarray:
        # Ensures DisCoPy's flat array is reshaped to a 2D (n, n) boolean array
        return np.array(self.hom.array, dtype=bool).reshape((self.n, self.n))

    def isbell_O(self, presheaf: np.ndarray) -> np.ndarray:
        """
        Computes O(P)_j = \\bigwedge_i (P_i => M_{ij}).
        """
        M = self._get_matrix()
        P = np.array(presheaf, dtype=bool)
        # Implication: (~P_i) OR M_{ij}
        implication = (~P[:, None]) | M
        return np.all(implication, axis=0)

    def isbell_Spec(self, copresheaf: np.ndarray) -> np.ndarray:
        """
        Computes Spec(Q)_i = \\bigwedge_j (Q_j => M_{ij}).
        """
        M = self._get_matrix()
        Q = np.array(copresheaf, dtype=bool)
        # Implication: (~Q_j) OR M_{ij}
        implication = (~Q[None, :]) | M
        return np.all(implication, axis=1)

    def isbell_completion(self) -> dict[tuple[str, ...], tuple[str, ...]]:
        """
        Computes the Isbell completion I(C) by generating all presheaves,
        taking their double dual Spec(O(P)), and collecting unique fixed points.
        """
        fixed_points = {}
        for num in range(1 << self.n):
            P = np.array([(num >> k) & 1 for k in range(self.n)], dtype=bool)

            Q = self.isbell_O(P)
            P_closed = self.isbell_Spec(Q)

            extent = tuple(obj.name for i, obj in enumerate(self.objects) if P_closed[i])
            intent = tuple(obj.name for i, obj in enumerate(self.objects) if Q[i])

            fixed_points[extent] = intent

        return fixed_points

# %%
# --- Test Run ---
names = ["6", "10", "15"]
hom = Matrix[bool].id(3)

C = EnrichedCategory(names, hom)
completion = C.isbell_completion()

for extent, intent in completion.items():
    print(f"Extent (P): {str(extent):<22} | Intent (Q): {intent}")

# %%
# Define the 3 objects in DisCoPy
names = ["6", "10", "15"]

# In DisCoPy, identity matrices represent discrete categories/antichains
hom = Matrix[bool].id(3)

# Construct the enriched category
C = EnrichedCategory(names, hom)

# Compute the Isbell completion
completion = C.isbell_completion()

for extent, intent in completion.items():
    print(f"Extent (P): {str(extent):<18} | Intent (Q): {intent}")


# %%
# --- Construct Divisors of 30 Poset ---
divisors = [1, 2, 3, 5, 6, 10, 15, 30]
names = [str(d) for d in divisors]
n = len(divisors)

# M_ij = True iff divisors[i] divides divisors[j]
adj_matrix = [
    [divisors[j] % divisors[i] == 0 for j in range(n)]
    for i in range(n)
]
flat_adj = [val for row in adj_matrix for val in row]

# Matrix[bool](array, dom, cod) takes standard ints n, n
hom = Matrix[bool](flat_adj, n, n)

# Compute Isbell Completion
C = EnrichedCategory(names, hom)
completion = C.isbell_completion()

# Display Results
print(f"{'Fixed Point (c)':<17} | {'Extent (P = ↓c)':<32} | {'Intent (Q = ↑c)'}")
print("-" * 80)
for extent, intent in completion.items():
    c = extent[-1] if extent else "1"
    print(f"c = {c:<13} | {str(extent):<32} | {intent}")


# %%
import numpy as np


class LawvereIsbell:
    def __init__(self, names: list[str], distance_matrix: np.ndarray):
        self.names = names
        self.n = len(names)
        self.D = np.array(distance_matrix, dtype=float)

    def isbell_O(self, presheaf: np.ndarray) -> np.ndarray:
        f = np.array(presheaf, dtype=float)
        return np.max(self.D - f[:, None], axis=0)

    def find_tight_span_point(self, f_init: np.ndarray, max_iter: int = 1000, tol: float = 1e-7) -> np.ndarray:
        f = np.array(f_init, dtype=float)
        for _ in range(max_iter):
            f_next = 0.5 * (f + self.isbell_O(f))
            if np.allclose(f, f_next, atol=tol):
                break
            f = f_next
        return np.round(f, decimals=4)

    def is_tight_span_element(self, f: np.ndarray, tol: float = 1e-4) -> bool:
        condition1 = np.all(f[:, None] + f[None, :] >= self.D - tol)
        f_dual = self.isbell_O(f)
        condition2 = np.allclose(f, f_dual, atol=tol)
        return condition1 and condition2

    def is_extremal_vertex(self, f: np.ndarray, tol: float = 1e-3) -> bool:
        """
        Checks if f is a 0D vertex of the tight span polyhedron.
        f is a vertex iff the matrix of tight constraints (f_i + f_j = d_ij) has rank n.
        """
        if not self.is_tight_span_element(f, tol=tol):
            return False

        # Build indicator matrix of tight edges where f_i + f_j == d_ij
        tight_mask = np.isclose(f[:, None] + f[None, :], self.D, atol=tol)

        # Build linear system rows for tight pairs
        rows = []
        for i in range(self.n):
            for j in range(i, self.n):
                if tight_mask[i, j]:
                    row = np.zeros(self.n)
                    row[i] += 1.0
                    row[j] += 1.0
                    rows.append(row)

        if not rows:
            return False

        # Vertex criterion: linear system has full rank n
        return np.linalg.matrix_rank(rows) == self.n

    def compute_tight_span_vertices(self, num_samples: int = 200) -> list[tuple]:
        unique_vertices = set()

        # 1. Base boundary candidates
        candidates = [self.D[i, :] for i in range(self.n)]

        # 2. Pairwise midpoints
        for i in range(self.n):
            for j in range(i + 1, self.n):
                candidates.append(0.5 * (self.D[i, :] + self.D[j, :]))

        # 3. Random convex samples
        rng = np.random.default_rng(42)
        for _ in range(num_samples):
            w = rng.dirichlet(np.ones(self.n))
            candidates.append(w @ self.D)

        # Map candidates to tight span points and filter for full-rank extremal vertices
        for f_init in candidates:
            f_fixed = self.find_tight_span_point(f_init)
            if self.is_extremal_vertex(f_fixed):
                unique_vertices.add(tuple(f_fixed))

        return sorted(list(unique_vertices))

# %%
# --- Test Case 1: Equilateral Triangle (3-point metric space) ---
# Side lengths = 2
names_tri = ["A", "B", "C"]
D_tri = np.array([
    [0, 2, 2],
    [2, 0, 2],
    [2, 2, 0]
])

metric_tri = LawvereIsbell(names_tri, D_tri)
vertices_tri = metric_tri.compute_tight_span_vertices()

print("=== 3-Point Metric Space (Equilateral Triangle) ===")
print("Tight Span Points f = (f_A, f_B, f_C):")
for v in vertices_tri:
    if metric_tri.is_tight_span_element(np.array(v)):
        print(f"  f = {v}  | Valid Tight Span Point? {metric_tri.is_tight_span_element(np.array(v))}")

print("\n" + "="*50 + "\n")

# %%
# --- Test Case 2: 4-Point Tree Metric ---
# Points A, B on left branch; C, D on right branch with central edge of length 1
# d(A,B)=2, d(C,D)=2, d(A,C)=4, d(A,D)=4, d(B,C)=4, d(B,D)=4
names_tree = ["A", "B", "C", "D"]
D_tree = np.array([
    [0, 2, 4, 4],
    [2, 0, 4, 4],
    [4, 4, 0, 2],
    [4, 4, 2, 0]
])

metric_tree = LawvereIsbell(names_tree, D_tree)
vertices_tree = metric_tree.compute_tight_span_vertices()

print("=== 4-Point Tree Metric Space ===")
print("Tight Span Points f = (f_A, f_B, f_C, f_D):")
for v in vertices_tree:
    if metric_tree.is_tight_span_element(np.array(v)):
        print(f"  f = {v}  | Valid Tight Span Point? {metric_tree.is_tight_span_element(np.array(v))}")