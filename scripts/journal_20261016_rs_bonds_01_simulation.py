import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    return mo, np, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # RuneScape Bond & RWT Macroeconomic Policy Simulator
    Interactive Stock-Flow simulation based on the **McDonnell Causal Graph (`RWT.md`)**[cite: 1].
    Adjust the policy levers below to observe the 180-day impulse response across membership cohorts, Grand Exchange Bond GP prices, and total cash revenue[cite: 1].
    """)
    return


@app.cell
def _(mo):
    # Policy Sliders for C-Level / Economy Designers
    rwt_ban_intensity = mo.ui.slider(
        0.01, 0.25, value=0.05, step=0.01, label="RWT & Bot Daily Ban Rate"
    )
    buyer_ban_deterrence = mo.ui.slider(
        0.0, 1.0, value=0.40, step=0.05, label="RWT Buyer Enforcement Share"
    )
    bond_cash_price_gbp = mo.ui.slider(
        4.99, 11.99, value=7.49, step=0.50, label="Bond Cash Price (£ GBP)"
    )
    ge_gold_sink_rate = mo.ui.slider(
        0.005, 0.05, value=0.015, step=0.005, label="GE Tax & Item Sink Intensity"
    )

    mo.hstack(
        [
            mo.vstack([rwt_ban_intensity, buyer_ban_deterrence]),
            mo.vstack([bond_cash_price_gbp, ge_gold_sink_rate]),
        ],
        justify="space-around",
    )
    return (
        bond_cash_price_gbp,
        buyer_ban_deterrence,
        ge_gold_sink_rate,
        rwt_ban_intensity,
    )


@app.cell
def _(
    bond_cash_price_gbp,
    buyer_ban_deterrence,
    ge_gold_sink_rate,
    np,
    pd,
    rwt_ban_intensity,
):
    # 180-Day Stock-Flow Differential Equations (Stabilized Log-Walrasian Tatonnement)
    days = 180
    dt = 1.0

    # Initial Stocks (Day 0 Baseline)
    cash_members = 850_000.0
    gp_members = 250_000.0
    bots = 120_000.0
    bond_price_gp = 11.4  # in Millions GP

    history = []

    for t in range(days):
        # 1. RWT Activity & Deterrence (Bots supply RWT; Buyer bans deter RWT demand)
        rwt_volume = (bots * 0.80) * (1.0 - 0.65 * buyer_ban_deterrence.value)

        # 2. Bond Demand in Gold (Grinders + Bots respond elastically to GE Bond GP price)
        grinder_gp_elasticity = (11.4 / bond_price_gp) ** 0.55
        bot_gp_elasticity = (11.4 / bond_price_gp) ** 0.35
        bond_demand_gp_units = (
            (gp_members / 14.0) * grinder_gp_elasticity
            + (bots / 14.0) * 0.85 * bot_gp_elasticity
        )

        # 3. Bond Supply from Cash Buyers (Calibrated to ~23k units/day at baseline)
        #    - Negatively linked to Cash Price (standard demand law)
        #    - Negatively linked to Bond GP Price (RWT.md: higher GP/bond = fewer bonds needed for target gear)
        #    - Positively linked to GE Item Sinks (preserves purchasing power of GP / high item value)
        #    - Negatively linked to RWT Volume (RWT acts as an illegal substitute for buying Bonds)
        price_elasticity_cash = (7.49 / bond_cash_price_gbp.value) ** 0.85
        gp_target_item_effect = (11.4 / bond_price_gp) ** 0.30 * (
            1.0 + 6.0 * (ge_gold_sink_rate.value - 0.015)
        )
        rwt_substitution_drag = np.exp(-0.0000035 * (rwt_volume - 57_600.0))

        bond_supply_cash_units = max(
            500.0,
            (cash_members * 0.027)
            * price_elasticity_cash
            * gp_target_item_effect
            * rwt_substitution_drag,
        )

        # 4. Stabilized GE Bond Price Adjustment (Log Excess Demand with tanh damping)
        log_excess_demand = np.log(
            max(bond_demand_gp_units, 1.0) / max(bond_supply_cash_units, 1.0)
        )
        daily_price_step = 0.08 * np.tanh(log_excess_demand) * dt
        bond_price_gp = float(
            np.clip(bond_price_gp * np.exp(daily_price_step), 4.0, 45.0)
        )

        # 5. Stock Updates: Bots, Cash Subscribers, and GP-Sustained Members
        bot_inflow = 6_000.0 * (max(rwt_volume, 100.0) / 57_600.0) ** 0.55
        bot_bans = bots * rwt_ban_intensity.value
        bots = max(2_000.0, bots + (bot_inflow - bot_bans) * dt)

        # Short-term RWT buyer bans (-) vs Long-term Integrity Retention boost (+)
        rwt_buyer_removals = (
            cash_members
            * 0.0012
            * rwt_ban_intensity.value
            * buyer_ban_deterrence.value
        )
        integrity_churn_rate = 0.0015 * (0.85 + 0.30 * (bots / 120_000.0))
        organic_acquisitions = 1_320.0 * (1.0 + 0.18 * (1.0 - bots / 120_000.0))

        cash_members = max(
            100_000.0,
            cash_members
            + (
                organic_acquisitions
                - cash_members * integrity_churn_rate
                - rwt_buyer_removals
            )
            * dt,
        )

        # Grinders adjust slowly based on Bond GP affordability
        grinder_net_growth = 0.0015 * ((11.4 / bond_price_gp) ** 0.35 - 1.0)
        gp_members = max(50_000.0, gp_members * (1.0 + grinder_net_growth * dt))

        daily_bond_rev_gbp = bond_supply_cash_units * bond_cash_price_gbp.value
        daily_sub_rev_gbp = (cash_members / 30.0) * 9.20

        history.append(
            {
                "Day": t,
                "Cash_Subscribers": cash_members,
                "GP_Bond_Members": gp_members,
                "Active_Bots": bots,
                "Bond_GE_Price_M_GP": bond_price_gp,
                "Daily_Bond_Cash_Units": bond_supply_cash_units,
                "Daily_Bond_Cash_Rev_GBP": daily_bond_rev_gbp,
                "Total_Daily_Revenue_GBP": daily_bond_rev_gbp + daily_sub_rev_gbp,
            }
        )

    df_sim = pd.DataFrame(history)
    return (df_sim,)


@app.cell
def _(df_sim, mo, plt):
    # Render 4-Panel Executive Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.patch.set_facecolor("#0f141c")
    for ax in axes.flat:
        ax.set_facecolor("#161f2c")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#cbd5e1")
        ax.yaxis.label.set_color("#cbd5e1")
        ax.title.set_color("#f1f5f9")
        for spine in ax.spines.values():
            spine.set_color("#2d3848")

    # Panel 1: Cash Subscribers vs GP Members
    axes[0, 0].plot(
        df_sim["Day"],
        df_sim["Cash_Subscribers"] / 1e3,
        color="#38bdf8",
        lw=2.5,
        label="Cash Sub ARR Cohort (k)",
    )
    axes[0, 0].plot(
        df_sim["Day"],
        df_sim["GP_Bond_Members"] / 1e3,
        color="#4ade80",
        lw=2.0,
        label="GP-Bond Sustained (k)",
    )
    axes[0, 0].set_title("Legitimate Member Cohorts (Thousands)")
    axes[0, 0].legend(facecolor="#131a24", labelcolor="#f1f5f9", fontsize=8)

    # Panel 2: Active Bots
    axes[0, 1].plot(
        df_sim["Day"], df_sim["Active_Bots"] / 1e3, color="#f87171", lw=2.5
    )
    axes[0, 1].set_title("Active RWT Bot Population (Thousands)")

    # Panel 3: Grand Exchange Bond Price (Millions GP)
    axes[1, 0].plot(
        df_sim["Day"], df_sim["Bond_GE_Price_M_GP"], color="#d4af37", lw=2.5
    )
    axes[1, 0].set_title("GE Bond Clearing Price (Millions GP)")
    axes[1, 0].set_xlabel("Simulation Day")

    # Panel 4: Daily Revenue (GBP)
    axes[1, 1].plot(
        df_sim["Day"],
        df_sim["Total_Daily_Revenue_GBP"] / 1e3,
        color="#4ade80",
        lw=2.5,
        label="Total Daily Rev (£k)",
    )
    axes[1, 1].plot(
        df_sim["Day"],
        df_sim["Daily_Bond_Cash_Rev_GBP"] / 1e3,
        color="#c084fc",
        lw=2.0,
        linestyle="--",
        label="Bond Cash Rev (£k)",
    )
    axes[1, 1].set_title("Daily Cash Bookings (£ GBP Thousands)")
    axes[1, 1].set_xlabel("Simulation Day")
    axes[1, 1].legend(facecolor="#131a24", labelcolor="#f1f5f9", fontsize=8)

    plt.tight_layout()

    # Summary KPI Delta Strip (Day 180 vs Day 0)
    d0, d180 = df_sim.iloc[0], df_sim.iloc[-1]
    rev_delta = (
        (d180["Total_Daily_Revenue_GBP"] - d0["Total_Daily_Revenue_GBP"])
        / d0["Total_Daily_Revenue_GBP"]
        * 100
    )
    gp_price_end = d180["Bond_GE_Price_M_GP"]
    bot_delta = (
        (d180["Active_Bots"] - d0["Active_Bots"]) / d0["Active_Bots"] * 100
    )

    kpi_strip = mo.hstack(
        [
            mo.stat(
                label="Day 180 GE Bond Price",
                value=f"{gp_price_end:.2f}M GP",
                caption="Equilibrium GE Price",
            ),
            mo.stat(
                label="180d Bot Population Delta",
                value=f"{bot_delta:+.1f}%",
                caption="RWT Supply Contraction",
            ),
            mo.stat(
                label="180d Total Daily Revenue Delta",
                value=f"{rev_delta:+.2f}%",
                caption="Subs + Cash Bonds (£ GBP)",
            ),
        ],
        justify="space-around",
    )

    mo.vstack([kpi_strip, fig])
    return


if __name__ == "__main__":
    app.run()
