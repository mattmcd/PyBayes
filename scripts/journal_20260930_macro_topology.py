"""
================================================================================
Macro-Financial Topological Simulation Engine (2026 Baseline Calibration)
================================================================================

Theoretical Foundation:
-----------------------
1. arXiv:2606.01663 (Hernandez & Sánchez-Soto):
   Spatiotemporal Cellular Sheaves & Cohomological Obstructions.
   - Models belief formation, multi-agent expectation alignment, and dynamic
     coordination via coupled sheaf diffusion: dx = (-alpha * L_0 * x + reactions) * dt.
   - Non-zero coboundary mismatch (||delta^0 x|| > 0) measures structural tension
     (H^1 obstruction) where local agent incentives clash with macro equilibrium.

2. arXiv:2503.17836 (Ghrist, Gould, Lopez, & Riess):
   Clearing Sections of Lattice Liability Networks.
   - Solves multi-agent financial settlement on complete product lattices [0, bar_p]
     using the Tarski-Kleene monotone fixed-point theorem.
   - Enforces legal debt priority (Senior Secured Debt > Subordinated / Junior Debt)
     under limited liability, preventing negative cash flows or phantom liquidity.

3. arXiv:2608.06020 (Han et al.):
   Economic World Models (EWM).
   - Multi-period simulation loop orchestrating decentralized agent reactions,
     verifiable macroeconomic closure, and physical balance-sheet invariants.

Agents / Economic Nodes (V):
----------------------------
- Node 0: Global Energy Desk / Oil Market (P_oil in $/bbl)
  Tracks global benchmark crude (Brent/WTI). Exogenous geopolitical shocks
  (e.g., Strait of Hormuz conflict / US-Iran sanctions) disrupt supply channels.
- Node 1: Federal Reserve / Monetary Authority (Fed Funds Target Rate in %)
  Dual-mandate central bank setting policy rates with quarterly inertia based
  on a Taylor rule anchored to the statutory 2.0% inflation target.
- Node 2: AI Hyperscalers / Tech Infra (Aggregate Quarterly CAPEX in $B)
  Capital-intensive compute infrastructure providers funded by a mix of operating
  cash flows, corporate bonds, and floating-rate private credit.
- Node 3: Non-Financial Enterprise / Corporate Sector (Operating Margin Index, 100 Base)
  Broad productive economy absorbing energy inputs, employing labor, and adopting
  AI compute to achieve cost/productivity efficiencies.
- Node 4: Banking & Financial Intermediary System (Reserves & Credit Buffers in $B)
  Senior and junior creditor holding liability claims on tech hyperscalers and
  commercial enterprises.

Feedback Mechanisms & Directed Edges (E):
-----------------------------------------
- Edge 0 (Oil -> Corporate): Energy Input Cost Pass-Through
  Spiking oil prices inflate transportation, utility, and raw material costs,
  compressing corporate operating margins.
- Edge 1 (Corporate -> Fed): Cost-Push Inflation & Taylor Reaction
  Margin compression and rising energy prices feed headline CPI/PCE inflation.
  When inflation exceeds the 2.0% target, the central bank tightens policy rates.
- Edge 2 (Fed -> Hyperscaler): Cost of Capital & Debt Service Drag
  Higher policy rates raise hurdle rates for speculative data-center CAPEX and
  inflate quarterly debt service burdens on floating-rate credit lines.
- Edge 3 (Hyperscaler -> Corporate): AI Productivity Disinflation
  Mature deployed compute clusters automate enterprise workflows, boosting corporate
  productivity and partially counteracting energy-induced margin compression.

Execution Cadence:
------------------
- Time Step: 1 Period = 1 Quarter (3 Months).
- Timeline: Q1 2026 (Calm baseline) through Q4 2028 (Normalization).
================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class DebtTranche:
    borrower_idx: int
    lender_idx: int
    senior_nominal: float   # Senior secured debt service ($B / quarter)
    junior_nominal: float   # Subordinated / private debt service ($B / quarter)


class LatticeClearingEngine:
    """
    Clearing Sections on Product Lattices (Ghrist et al., arXiv:2503.17836).
    Enforces debt seniority under limited liability via Tarski fixed-point iteration.
    """
    def __init__(self, num_agents: int, tranches: List[DebtTranche]):
        self.num_agents = num_agents
        self.tranches = tranches

    def compute_clearing(
        self,
        operating_cash_flows: np.ndarray,
        liquid_reserves: np.ndarray,
        tol: float = 1e-6,
        max_iter: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_tranches = len(self.tranches)
        p_sen = np.array([t.senior_nominal for t in self.tranches], dtype=float)
        p_jun = np.array([t.junior_nominal for t in self.tranches], dtype=float)

        for _ in range(max_iter):
            p_sen_prev, p_jun_prev = p_sen.copy(), p_jun.copy()
            incoming = np.zeros(self.num_agents)
            for idx, t in enumerate(self.tranches):
                incoming[t.lender_idx] += p_sen[idx] + p_jun[idx]

            available_funds = np.maximum(0.0, operating_cash_flows + liquid_reserves + incoming)

            for idx, t in enumerate(self.tranches):
                cash = available_funds[t.borrower_idx]
                # Senior claims receive priority distribution
                p_sen[idx] = min(t.senior_nominal, cash)
                residual = max(0.0, cash - t.senior_nominal)
                # Junior/subordinated claims absorb residual capacity
                p_jun[idx] = min(t.junior_nominal, residual)

            if np.max(np.abs(p_sen - p_sen_prev)) < tol and np.max(np.abs(p_jun - p_jun_prev)) < tol:
                break

        sen_defaults = np.array([t.senior_nominal - p_sen[i] for i, t in enumerate(self.tranches)])
        jun_defaults = np.array([t.junior_nominal - p_jun[i] for i, t in enumerate(self.tranches)])
        return p_sen, p_jun, sen_defaults + jun_defaults


class CenteredMacroSheaf:
    """
    Cellular Sheaf over the Macro Coordination Complex (Hernandez & Sánchez-Soto, arXiv:2606.01663).
    Evaluates expectations on deviations from the stationary 2026 baseline.
    """
    def __init__(self, base_state: np.ndarray):
        self.num_nodes = 5
        self.base_state = base_state.copy()
        self.edges = [(0, 3), (3, 1), (1, 2), (2, 3)]
        self._build_coboundary_operator()

    def _build_coboundary_operator(self):
        # 0-coboundary operator delta^0: C^0 -> C^1
        self.delta_0 = np.zeros((len(self.edges), self.num_nodes))

        # Edge 0 (Oil -> Margin): +$10/bbl crude dev trims ~0.8 margin points
        self.delta_0[0, 0] =  0.08
        self.delta_0[0, 3] =  1.00

        # Edge 1 (Margin -> Fed): Cost-push compression induces hawkish tension
        self.delta_0[1, 3] = -0.15
        self.delta_0[1, 1] =  1.00

        # Edge 2 (Fed -> CAPEX): Policy rate hike dampens uncommitted AI capital allocation
        self.delta_0[2, 1] =  0.80
        self.delta_0[2, 2] =  0.25

        # Edge 3 (CAPEX -> Margin): Mature AI deployment feeds enterprise productivity
        self.delta_0[3, 2] = -0.05
        self.delta_0[3, 3] =  1.00

        # Sheaf 0-Laplacian: L_0 = (delta^0)^T delta^0
        self.L_0 = self.delta_0.T @ self.delta_0

    def compute_mismatch(self, states: np.ndarray) -> np.ndarray:
        deviations = states - self.base_state
        return self.delta_0 @ deviations

    def diffuse(self, states: np.ndarray, reactions: np.ndarray, dt: float = 0.25) -> np.ndarray:
        deviations = states - self.base_state
        alpha = 0.05  # Coordination diffusion rate
        d_dev = -alpha * (self.L_0 @ deviations) + reactions
        return states + d_dev * dt


class Calibrated2026Economy:
    """
    Macroeconomic Simulation Environment (Han et al., arXiv:2608.06020).
    Coordinates quarterly feedback loops, policy reactions, and financial settlement.
    """
    def __init__(self):
        # Statutory central bank & macro policy targets
        self.inflation_target = 2.00     # Fed PCE target (pi*)
        self.start_inflation = 2.60      # Real-world headline baseline (Q1 2026)
        self.neutral_real_rate = 1.375   # Real neutral rate (r*)
        self.start_fed_rate = 3.875      # Nominal policy baseline midpoint

        self.taylor_phi_pi = 0.60        # Weight on inflation gap (pi - pi*)
        self.taylor_phi_y = 0.35         # Weight on output/margin gap

        # Q1 2026 Baseline State Vector:
        # [Oil: $75, Fed Rate: 3.875%, AI CAPEX: $170B/qtr, Margin Index: 100.0, Reserves: $300B]
        self.baseline = np.array([75.0, self.start_fed_rate, 170.0, 100.0, 300.0], dtype=float)
        self.state = self.baseline.copy()
        self.sheaf = CenteredMacroSheaf(self.baseline)

        # Debt Service Commitments ($B / quarter):
        # - Hyperscalers: $28B Senior, $12B Subordinated/Private Credit
        # - Non-Financial Corporates: $45B Senior, $15B Subordinated/Revolvers
        self.tranches = [
            DebtTranche(borrower_idx=2, lender_idx=4, senior_nominal=28.0, junior_nominal=12.0),
            DebtTranche(borrower_idx=3, lender_idx=4, senior_nominal=45.0, junior_nominal=15.0),
        ]
        self.clearing_engine = LatticeClearingEngine(5, self.tranches)

    def run_quarter(self, q_label: str, oil_shock_rate: float) -> Dict[str, float]:
        p_oil, fed_rate_cont, ai_capex, corp_margin, _ = self.state

        oil_dev = p_oil - 75.0
        margin_dev = corp_margin - 100.0

        # 1. Macro Inflation Dynamics
        oil_pass_through = 0.022 * oil_dev
        ai_disinflation = -0.004 * (ai_capex - 170.0)
        current_inflation = max(1.5, self.start_inflation + oil_pass_through + ai_disinflation)

        inflation_gap = current_inflation - self.inflation_target
        output_gap = (margin_dev / 100.0) * 1.5

        # Taylor Rule target: i* = r* + pi + phi_pi*(pi - pi*) + phi_y*y
        taylor_target = (
            self.neutral_real_rate
            + current_inflation
            + self.taylor_phi_pi * inflation_gap
            + self.taylor_phi_y * output_gap
        )

        # Central Bank adjustment with quarterly policy inertia
        fed_reaction = 0.65 * (taylor_target - fed_rate_cont)

        # Hyperscaler CAPEX momentum vs interest rate drag
        capex_organic = 1.6
        rate_drag = -3.5 * (fed_rate_cont - self.start_fed_rate)
        ai_reaction = capex_organic + rate_drag

        # Corporate margins: energy cost drag vs compute productivity gains
        corp_reaction = -0.07 * oil_dev + 0.05 * (ai_capex - 170.0)

        reactions = np.array([oil_shock_rate, fed_reaction, ai_reaction, corp_reaction, 0.0])

        # 2. Continuous Sheaf Diffusion
        self.state = self.sheaf.diffuse(self.state, reactions, dt=0.25)
        p_oil, fed_rate_cont, ai_capex, corp_margin, _ = self.state

        # Snap policy rate to 25 bps increments for settlement and public reporting
        fed_rate_snapped = np.round(fed_rate_cont * 4.0) / 4.0

        # 3. Cash Flow Determination & Lattice Settlement ($B / quarter)
        rate_burden = (fed_rate_snapped / self.start_fed_rate) * 14.0
        hyp_cf = 44.0 + 0.08 * (ai_capex - 170.0) - rate_burden
        corp_cf = 68.0 * (corp_margin / 100.0) - 0.12 * oil_dev

        cf_vector = np.array([0.0, 0.0, hyp_cf, corp_cf, 0.0])
        reserves = np.array([0.0, 0.0, 18.0, 22.0, 300.0])

        p_sen, p_jun, defaults = self.clearing_engine.compute_clearing(cf_vector, reserves)
        h1_norm = np.linalg.norm(self.sheaf.compute_mismatch(self.state))

        return {
            "quarter": q_label,
            "oil_price": p_oil,
            "inflation_pct": current_inflation,
            "fed_rate": fed_rate_snapped,
            "ai_capex": ai_capex,
            "corp_margin": corp_margin,
            "tech_default": defaults[0],
            "h1_obstruction": h1_norm
        }


# ============================================================================
# EXECUTION DEMO: 2026–2028 Geopolitical Oil Shock Progression
# ============================================================================

if __name__ == "__main__":
    sim = Calibrated2026Economy()

    quarters = [
        ("Q1 2026",  0.0),   # Baseline ($75 Brent, 2.6% inflation, 4.00% Fed rate)
        ("Q2 2026", 45.0),   # US/Iran geopolitical flare-up, Hormuz risk pricing
        ("Q3 2026", 35.0),   # Peak disruption & supply bottlenecks
        ("Q4 2026", 10.0),   # High plateau, SPR release / partial rerouting
        ("Q1 2027", -15.0),  # De-escalation & non-OPEC production adjustments
        ("Q2 2027", -25.0),  # Normalization toward long-term trend
        ("Q3 2027", -15.0),
        ("Q4 2027",  -5.0),
        ("Q1 2028",   0.0),
        ("Q2 2028",   0.0),
        ("Q3 2028", 0.0),
        ("Q4 2028", 0.0),
    ]

    print(f"{'Quarter':<9} | {'Brent Oil':<11} | {'Inflation':<10} | {'Fed Rate':<9} | {'AI CAPEX':<12} | {'Corp Margin':<12} | {'Tech Def':<9} | {'H1 Norm':<8}")
    print("-" * 94)

    for q_label, shock in quarters:
        m = sim.run_quarter(q_label, shock)
        print(
            f"{m['quarter']:<9} | "
            f"${m['oil_price']:<10.2f} | "
            f"{m['inflation_pct']:<8.2f}% | "
            f"{m['fed_rate']:<8.2f}% | "
            f"${m['ai_capex']:<10.2f}B | "
            f"{m['corp_margin']:<12.2f} | "
            f"${m['tech_default']:<7.2f}B | "
            f"{m['h1_obstruction']:<8.3f}"
        )