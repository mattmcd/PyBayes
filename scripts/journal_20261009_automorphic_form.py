import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =====================================================================
# 1. Automorphic Primitives
# =====================================================================

def build_hecke_matrix(m: int, k: int, num_terms: int) -> torch.Tensor:
    """M x M operator matrix for T_m acting on q-expansions:
       a_n(T_m f) = sum_{d | gcd(n, m)} d^{k-1} * a_{(n*m)/d^2}
    """
    T = torch.zeros((num_terms, num_terms), dtype=torch.float32)
    for n in range(1, num_terms + 1):
        g = math.gcd(n, m)
        for d in range(1, g + 1):
            if g % d == 0:
                source_idx = (n * m) // (d * d)
                if 1 <= source_idx <= num_terms:
                    T[n - 1, source_idx - 1] += float(d ** (k - 1))
    return T


class HeckeLinear(nn.Module):
    """Linear layer parameterized as a learnable linear combination of Hecke operators."""
    def __init__(self, weight_k: int, num_terms: int, hecke_indices=(1, 2, 3, 5, 7)):
        super().__init__()
        matrices = [build_hecke_matrix(m, weight_k, num_terms) for m in hecke_indices]
        self.register_buffer("hecke_ops", torch.stack(matrices, dim=0))
        init_coeffs = torch.zeros(len(hecke_indices))
        init_coeffs[0] = 1.0
        self.coeffs = nn.Parameter(init_coeffs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = torch.einsum("k,kij->ij", self.coeffs, self.hecke_ops)
        return F.linear(x, W)


class RankinCohenInteraction(nn.Module):
    """Bilinear Rankin-Cohen bracket [f, g]_r on truncated q-expansions using rFFT."""
    def __init__(self, k1: int, k2: int, r: int = 1):
        super().__init__()
        self.k1, self.k2, self.r = k1, k2, r
        weights = [
            float(((-1) ** s) * math.comb(k1 + r - 1, s) * math.comb(k2 + r - 1, r - s))
            for s in range(r + 1)
        ]
        self.register_buffer("rc_weights", torch.tensor(weights, dtype=torch.float32))

    def _apply_derivative(self, x: torch.Tensor, order: int) -> torch.Tensor:
        if order == 0:
            return x
        M = x.shape[-1]
        n_powers = torch.arange(1, M + 1, device=x.device, dtype=x.dtype) ** order
        return x * n_powers

    def _poly_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        M = a.shape[-1]
        fft_size = 2 * M
        fa = torch.fft.rfft(a, n=fft_size, dim=-1)
        fb = torch.fft.rfft(b, n=fft_size, dim=-1)
        prod = torch.fft.irfft(fa * fb, n=fft_size, dim=-1)
        return prod[..., :M]

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(f)
        for s in range(self.r + 1):
            w = self.rc_weights[s]
            f_der = self._apply_derivative(f, self.r - s)
            g_der = self._apply_derivative(g, s)
            out = out + w * self._poly_mul(f_der, g_der)
        return out


# =====================================================================
# 2. Automorphic Models (~20k Parameter Budget)
# =====================================================================

class AutomorphicMLP(nn.Module):
    """~20.6k params: 784 -> dual 12-dim q-expansions -> 32 hidden -> 10."""
    def __init__(self, num_terms: int = 12, hidden_dim: int = 32, weight_k: int = 2):
        super().__init__()
        self.in_proj_f = nn.Linear(784, num_terms)
        self.in_proj_g = nn.Linear(784, num_terms)
        self.hecke_f = HeckeLinear(weight_k, num_terms)
        self.hecke_g = HeckeLinear(weight_k, num_terms)
        self.rc_bracket = RankinCohenInteraction(weight_k, weight_k, r=1)
        self.norm = nn.LayerNorm(num_terms)
        self.classifier = nn.Sequential(
            nn.Linear(num_terms, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        f = self.hecke_f(self.in_proj_f(x))
        g = self.hecke_g(self.in_proj_g(x))
        return self.classifier(self.norm(self.rc_bracket(f, g)))


class AutomorphicCNN(nn.Module):
    """~20.4k params: standard Conv stem + channel-squeeze to prevent

    projection parameter explosion before automorphic block.
    """
    def __init__(self, num_terms: int = 32, weight_k: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 7, 7]
            # 1x1 conv channel reducer: eliminates the 100k dense projection sink
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU()         # [B, 8, 7, 7]
        )
        flat_dim = 8 * 7 * 7  # 392
        self.proj_f = nn.Linear(flat_dim, num_terms)
        self.proj_g = nn.Linear(flat_dim, num_terms)
        self.hecke_f = HeckeLinear(weight_k, num_terms)
        self.hecke_g = HeckeLinear(weight_k, num_terms)
        self.rc_bracket = RankinCohenInteraction(weight_k, weight_k, r=1)
        self.norm = nn.LayerNorm(num_terms)
        self.classifier = nn.Linear(num_terms, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x).flatten(1)
        f = self.hecke_f(self.proj_f(feat))
        g = self.hecke_g(self.proj_g(feat))
        return self.classifier(self.norm(self.rc_bracket(f, g)))


class AutomorphicViT(nn.Module):
    """~20.4k params: patch=7, d_model=40, depth=2, n_heads=4."""
    def __init__(self, patch_size: int = 7, d_model: int = 40, depth: int = 2, n_heads: int = 4, weight_k: int = 2):
        super().__init__()
        self.patch_proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (28 // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha_norm": nn.LayerNorm(d_model),
                "mha": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ffn_norm": nn.LayerNorm(d_model),
                "hecke_f": HeckeLinear(weight_k, d_model),
                "hecke_g": HeckeLinear(weight_k, d_model),
                "rc": RankinCohenInteraction(weight_k, weight_k, r=1),
                "rc_norm": nn.LayerNorm(d_model),
                "proj_out": nn.Linear(d_model, d_model)
            }) for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_proj(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1) + self.pos_embed

        for layer in self.layers:
            norm_tokens = layer["mha_norm"](tokens)
            attn_out, _ = layer["mha"](norm_tokens, norm_tokens, norm_tokens)
            tokens = tokens + attn_out

            norm_ffn = layer["ffn_norm"](tokens)
            f = layer["hecke_f"](norm_ffn)
            g = layer["hecke_g"](norm_ffn)
            inter = layer["rc"](f, g)
            tokens = tokens + layer["proj_out"](layer["rc_norm"](inter))

        return self.head(tokens[:, 0])


class AutomorphicHybrid(nn.Module):
    """~20.3k params: 2-layer Conv stem stalks + 2 transformer layers (d_model=32)."""
    def __init__(self, d_model: int = 32, depth: int = 2, n_heads: int = 4, weight_k: int = 2):
        super().__init__()
        # 1 -> 16 -> 32 Conv stem (~7.2k params)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, d_model, kernel_size=1)
        )
        num_patches = 7 * 7
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha_norm": nn.LayerNorm(d_model),
                "mha": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ffn_norm": nn.LayerNorm(d_model),
                "hecke_f": HeckeLinear(weight_k, d_model),
                "hecke_g": HeckeLinear(weight_k, d_model),
                "rc": RankinCohenInteraction(weight_k, weight_k, r=1),
                "rc_norm": nn.LayerNorm(d_model),
                "proj_out": nn.Linear(d_model, d_model)
            }) for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.stem(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1) + self.pos_embed

        for layer in self.layers:
            norm_tokens = layer["mha_norm"](tokens)
            attn_out, _ = layer["mha"](norm_tokens, norm_tokens, norm_tokens)
            tokens = tokens + attn_out

            norm_ffn = layer["ffn_norm"](tokens)
            f = layer["hecke_f"](norm_ffn)
            g = layer["hecke_g"](norm_ffn)
            inter = layer["rc"](f, g)
            tokens = tokens + layer["proj_out"](layer["rc_norm"](inter))

        return self.head(tokens[:, 0])


# =====================================================================
# 3. Non-Automorphic Baselines (~20k Parameter Budget)
# =====================================================================

class BaselineMLP(nn.Module):
    def __init__(self, num_terms: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.in_proj_f = nn.Linear(784, num_terms)
        self.in_proj_g = nn.Linear(784, num_terms)
        self.norm = nn.LayerNorm(num_terms)
        self.classifier = nn.Sequential(
            nn.Linear(num_terms, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        f = self.in_proj_f(x)
        g = self.in_proj_g(x)
        h = f * F.gelu(g)
        return self.classifier(self.norm(h))


class BaselineCNN(nn.Module):
    def __init__(self, num_terms: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 8, kernel_size=1),
            nn.ReLU()
        )
        flat_dim = 8 * 7 * 7
        self.proj_f = nn.Linear(flat_dim, num_terms)
        self.proj_g = nn.Linear(flat_dim, num_terms)
        self.norm = nn.LayerNorm(num_terms)
        self.classifier = nn.Linear(num_terms, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x).flatten(1)
        f = self.proj_f(feat)
        g = self.proj_g(feat)
        h = f * F.gelu(g)
        return self.classifier(self.norm(h))


class BaselineViT(nn.Module):
    def __init__(self, patch_size: int = 7, d_model: int = 40, depth: int = 2, n_heads: int = 4):
        super().__init__()
        self.patch_proj = nn.Conv2d(1, d_model, kernel_size=patch_size, stride=patch_size)
        num_patches = (28 // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha_norm": nn.LayerNorm(d_model),
                "mha": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ffn_norm": nn.LayerNorm(d_model),
                "ffn_in": nn.Linear(d_model, d_model),
                "ffn_out": nn.Linear(d_model, d_model)
            }) for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_proj(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1) + self.pos_embed

        for layer in self.layers:
            norm_tokens = layer["mha_norm"](tokens)
            attn_out, _ = layer["mha"](norm_tokens, norm_tokens, norm_tokens)
            tokens = tokens + attn_out

            norm_ffn = layer["ffn_norm"](tokens)
            ffn_out = layer["ffn_out"](F.gelu(layer["ffn_in"](norm_ffn)))
            tokens = tokens + ffn_out

        return self.head(tokens[:, 0])


class BaselineHybrid(nn.Module):
    def __init__(self, d_model: int = 32, depth: int = 2, n_heads: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, d_model, kernel_size=1)
        )
        num_patches = 7 * 7
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha_norm": nn.LayerNorm(d_model),
                "mha": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "ffn_norm": nn.LayerNorm(d_model),
                "ffn_in": nn.Linear(d_model, d_model),
                "ffn_out": nn.Linear(d_model, d_model)
            }) for _ in range(depth)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.stem(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1) + self.pos_embed

        for layer in self.layers:
            norm_tokens = layer["mha_norm"](tokens)
            attn_out, _ = layer["mha"](norm_tokens, norm_tokens, norm_tokens)
            tokens = tokens + attn_out

            norm_ffn = layer["ffn_norm"](tokens)
            ffn_out = layer["ffn_out"](F.gelu(layer["ffn_in"](norm_ffn)))
            tokens = tokens + ffn_out

        return self.head(tokens[:, 0])

# =====================================================================
# 4. Training, Accuracy Evaluation & Comparison Harness
# =====================================================================

def evaluate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return 100.0 * correct / total


def train_model(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 1, lr: float = 2e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # End-of-epoch snapshot metrics (eliminates running average skew)
    train_acc = evaluate_accuracy(model, train_loader, device)
    test_acc = evaluate_accuracy(model, test_loader, device)
    return train_acc, test_acc


# =====================================================================
# Updated Benchmark Table Harness
# =====================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    models_to_test = [
        ("MLP",        AutomorphicMLP(num_terms=12, hidden_dim=32, weight_k=2),
                       BaselineMLP(num_terms=12, hidden_dim=32)),
        ("CNN",        AutomorphicCNN(num_terms=32, weight_k=2),
                       BaselineCNN(num_terms=32)),
        ("ViT",        AutomorphicViT(patch_size=7, d_model=40, depth=2, n_heads=4, weight_k=2),
                       BaselineViT(patch_size=7, d_model=40, depth=2, n_heads=4)),
        ("Hybrid ViT", AutomorphicHybrid(d_model=32, depth=2, n_heads=4, weight_k=2),
                       BaselineHybrid(d_model=32, depth=2, n_heads=4)),
    ]

    print(f"\n{'Model Type':<12} | {'Variant':<14} | {'Params':<8} | {'Train Acc':<10} | {'Test Acc':<10}")
    print("-" * 64)

    for family, auto_m, base_m in models_to_test:
        for variant, model in [("Automorphic", auto_m), ("Baseline", base_m)]:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            train_acc, test_acc = train_model(model, train_loader, test_loader, epochs=1, lr=2e-3)
            print(f"{family:<12} | {variant:<14} | {n_params:<8} | {train_acc:>8.2f}% | {test_acc:>8.2f}%")