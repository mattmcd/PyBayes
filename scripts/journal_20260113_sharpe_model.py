# %%
from IPython.core.pylabtools import figsize

from startup import np, pd, plt, sns, sm, smf, Path
import sqlalchemy as sa
from xarray import DataArray
# %%
import pymc as pm
import bambi as bmb
import arviz as az

# %%
from mattmcd.io import pg_engine

# %%
import pytensor
pytensor.config.cxx = '' # Fix linker errors by reverting to Python '/usr/bin/clang++'

# %%
engine = pg_engine()
metadata = sa.MetaData()

# %%
# 2025-12-19 New price tables
tb_fn = sa.Table('return_model', metadata, autoload_with=engine, schema='raw')
tb_fn_cte = sa.select(
    tb_fn.c.date,
    tb_fn.c.ticker,
    tb_fn.c.pct_change_gbp.label('pct_change')
).where(~tb_fn.c.ticker.in_(['CLDN'])).cte('filtered_ftse')

# %%
# src_table = query_pct_change(tb_f)
src_table = sa.select(tb_fn_cte).order_by(tb_fn_cte.c.ticker, tb_fn_cte.c.date)

df_rf = pd.read_sql(src_table, engine).assign(
    date=lambda df: pd.to_datetime(df.date)
).set_index(['date', 'ticker']).unstack()

# %%
# 21 day rolling annualized Sharpe ratio
df_s = df_rf.xs('pct_change', level=0, axis=1).fillna(0).rolling(21).apply(
    lambda x: x.mean()/x.std()*np.sqrt(252)
)

# %%
# Index constituents for name lookup
tb_ic = sa.Table('index_constituents', metadata, schema='equity', autoload_with=engine)
df_ic = pd.read_sql(sa.select(tb_ic), engine)

# %%
# Mean rolling Sharpe ratio over last 21 days
class SharpeDataset:
    def __init__(self, df_stock_returns, df_index_constituents):
        self.df_s = df_stock_returns
        self.df_ic = df_index_constituents

    def rolling_sharpe(self, date=None, idx=None):
        df_date = self.df_s.rolling(21).mean().loc[date] if date is not None else self.df_s.rolling(21).mean().iloc[idx]
        df_top = pd.merge(
            df_date.sort_values(ascending=False),
            self.df_ic.loc[:, ['ticker', 'company', 'sector']].set_index('ticker'),
            right_index=True, left_index=True, how='left'
        )
        df_top.columns=['sharpe', 'company', 'sector']

        # Normalize sector
        df_top['sector'] = df_top.sector.str.title().str.replace(
            ' And ', ' & '
        ).str.replace(
            'Telecomms', 'Telecommunications'
        ).str.replace(
            'Nonlife', 'Non-Life'
        )
        return df_top

    @property
    def latest_date(self):
        return self.df_s.index[-1].date().isoformat()

    def latest_rolling_sharpe(self):
        return self.rolling_sharpe(date=self.latest_date)

# %%
def plot_sharpe_by_sector(df_top, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 15))
        do_plot = True
    else:
        fig = ax.figure
        do_plot = False
    df_sector_plot = df_top
    # df_sector_plot = df_top.loc[df_top.sector.isin(most_common_sectors), :]
    sns.boxplot(df_sector_plot, y='sector', x='sharpe', ax=ax)
    ax.axvline(0, c='k', ls='--')
    ax.set_title('Sharpe Ratio by Sector')
    if do_plot:
        plt.tight_layout()
        plt.show()
    return fig

# %%
ds_s = SharpeDataset(df_s, df_ic)

# %%
data = ds_s.latest_rolling_sharpe().reset_index().dropna()
asof_date = ds_s.latest_date
# %%
# Code from Google AI Pro
# 2. Build the Hierarchical Student-t Model
# We model the 'mu' (mean) of the distribution
# 'nu' (degrees of freedom) controls the fatness of the tails
model_heavy = bmb.Model(
    "sharpe ~ 1 + (1|sector)", # + (1|sector:ticker)",
    data,
    family="t"
)

# %%
model_heavy.build()

# %%
# 3. Fit the model
# We increase 'target_accept' because ratio distributions can be tricky to sample
results_heavy = model_heavy.fit(
    draws=2000, tune=1000, target_accept=0.98, nuts_sampler='blackjax'
)


# %%
# 4. View the "nu" parameter (Degrees of Freedom)
# A low nu (~3-5) confirms the "Normal/Half-Normal" ratio behavior
df_res = az.summary(results_heavy, var_names=['1|sector']).sort_values('hdi_97%', ascending=False)
print(df_res)

# %%
def sort_sectors(results_heavy):
    # 1. Calculate the median of the '1|sector' offsets from the posterior
    # We take the mean across chains/draws for each sector
    posterior = results_heavy.posterior["1|sector"]
    medians = posterior.median(dim=["chain", "draw"])

    # 2. Get the sector names and sort them by the median values
    sector_names = results_heavy.posterior["sector__factor_dim"].values
    sorted_indices = medians.argsort().values
    sorted_sectors = sector_names[sorted_indices][::-1]
    return sorted_sectors

# %%

def plot_ridgeplot(res, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 15))
        do_plot = True
    else:
        fig = ax.figure
        do_plot = False

    sorted_sectors = sort_sectors(res)
    az.plot_forest(
        res,
        kind='ridgeplot',
        coords={"sector__factor_dim": sorted_sectors},  # This enforces the sort order
        var_names=["1|sector"],
        combined=True,
        colors="blue",
        ax=ax,
        ridgeplot_overlap=1.5,
        ridgeplot_alpha=0.8,
    )
    ax.axvline(-res.posterior['Intercept'].median(dim=["chain", "draw"]), color='black', linestyle='--')
    ax.axvline(0, color='red', linestyle='--')
    ax.set_title('Hierarchical Model of Sharpe Ratio by Sector')
    if do_plot:
        plt.tight_layout()
        plt.show()
    return fig

# %%
def plot_combined(df, res, asof_date):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))
    plot_sharpe_by_sector(df, axs[0])
    plot_ridgeplot(res, axs[1])
    fig.suptitle(f'{asof_date}')
    plt.tight_layout()
    plt.show()
    return fig

# %%
fig = plot_combined(data, results_heavy, asof_date)

# %%
# Combine into a class
class SectorSharpeModel:
    def __init__(self, sd : SharpeDataset, date : str):
        self.df = sd.rolling_sharpe(date=date)
        self.data = None
        self.date = date
        self.model = None
        self.res = None

    def preprocess(self):
        self.data = self.df.reset_index().dropna()

    def build(self):
        self.preprocess()
        self.model = bmb.Model(
            "sharpe ~ 1 + (1|sector)",  # + (1|sector:ticker)",
            self.data,
            family="t"
        )
        self.model.build()
        self.res = self.model.fit(
            draws=2000, tune=1000, target_accept=0.98, nuts_sampler='blackjax'
        )
        return self

    def plot_combined(self):
        fig = plot_combined(self.data, self.res, self.date)
        return fig

# %%
model = SectorSharpeModel(ds_s, ds_s.latest_date).build()
model.plot_combined()
# %%
# Get last business date in each quarter
quarter_end_dates = [
    ts.date().isoformat()
    for ts in ds_s.df_s.assign(
        start_quarter_date=lambda df: df.index.to_period('Q').to_timestamp('D')
    ).reset_index().groupby('start_quarter_date')['date'].max().tolist()
]


# %%
plot_dir = Path.home() / 'Work' / 'Presentations'/ '2026Q1_Sharpe_Sector_Plots'

# %%
def fit_models_by_dates(sd : SharpeDataset, dates : list[str]):
    models = []
    for date in dates:
        model = SectorSharpeModel(sd, date).build()
        models.append((date, model))

    models = dict(models)
    return models

# %%
models = fit_models_by_dates(ds_s, quarter_end_dates)

# %%
for date, model in models.items():
    fig = model.plot_combined()
    fig.savefig(plot_dir / f'sector_dispersion_{date}.png')

# %%
# Overall Sharpe ratio
def get_sharpe_ratio(models):
    df_sr = pd.DataFrame(
        {
            'date': models.keys(),
            'sharpe':[
                m.res.posterior['Intercept'].median(dim=["chain", "draw"]).to_numpy() for m in models.values()
            ],
            'sector_std':[
                m.res.posterior['1|sector'].median(dim=["chain", "draw"]).std().to_numpy() for m in models.values()
            ],
            'sector_median':[
                m.res.posterior['1|sector'].median(dim=["chain", "draw"]).median().to_numpy() for m in models.values()
            ]
        }
    ).assign(
        date=lambda x: pd.to_datetime(x['date'])
    ).set_index('date').astype(float)
    return df_sr

# %%
# Plot overall Sharpe ratio
df_sr = get_sharpe_ratio(models)
ax = df_sr.sharpe.plot(marker='o', figsize=(10, 5))
ax.axhline(0, color='k', ls='--')
plt.show()

# %%
# Increase resolution to monthly
month_end_dates = [
    ts.date().isoformat()
    for ts in ds_s.df_s.assign(
        start_month_date=lambda df: df.index.to_period('M').to_timestamp('D')
    ).reset_index().groupby('start_month_date')['date'].max().tolist()
][1:]

# %%
models_me = fit_models_by_dates(ds_s, month_end_dates)

# %%
df_sr = get_sharpe_ratio(models_me)

df_sr_perf = df_sr.assign(
    is_market_up=lambda df: df.sharpe > 0,
    is_best_sector_up=lambda df: (df.sharpe + df.sector_median + 1.67*df.sector_std) > 0,
    # is_both_up=lambda df: df.is_market_up & df.is_best_sector_up
).loc[:, ['is_market_up', 'is_best_sector_up',]].agg(
    ['mean', 'sum']
).round(2)

print(df_sr_perf)

# %%
# Plot overall Sharpe ratio and sector dispersion
# Hypothesis: The Sharpe line is a measure of risk-adjusted return, while the sector dispersion bands
# show what can be achieved by sector selection.
# For example in March 2020 sector selection wouldn't have helped much but there
# are periods where selecting the right sector can result in positive returns even
# though the overall market is down.
def plot_sharpe_dispersion(df_sr, ax=None):
    ax.plot(df_sr.index, df_sr.sharpe, marker='o', label='Market Sharpe Ratio')
    ax.plot(df_sr.index, df_sr.sector_median, )
    ax.plot(
        df_sr.index,
        df_sr.sector_median + 1.67*df_sr.sector_std + df_sr.sharpe,
        color='red', alpha=0.5, marker='.', linestyle=':', label='Best Sector'
    )
    ax.plot(
        df_sr.index,
        df_sr.sector_median - 1.67*df_sr.sector_std + df_sr.sharpe,
        color='red', alpha=0.5, marker='.', linestyle=':', label='Worst Sector'
    )
    ax.legend()
    ax.axhline(0, color='k', ls='--')
    ax.set_title(
        'Sharpe Ratio vs. Sector Dispersion\n'
        f'Monthly hit rate: Market {df_sr_perf.loc["mean", "is_market_up"]:.0%} / Best Sector {df_sr_perf.loc["mean", "is_best_sector_up"]:.0%}'
    )

# %%
fig, ax = plt.subplots(figsize=(10, 5))
plot_sharpe_dispersion(df_sr, ax=ax)
plt.show()

fig.savefig(plot_dir / 'sector_dispersion_monthly.png')

# %%
df_sr.sharpe.hist()
plt.show()

# %%
df_sr.sector_std.hist()
plt.show()

# %%
sns.scatterplot(
    df_sr, x='sharpe', y='sector_std', alpha=0.5
)
plt.show()

# %%
# Sector dispersion as a leading indicator?  Not really
smf.ols(
    'sharpe ~ sector_std_lag',
    df_sr.assign(sector_std_lag=lambda df: df.sector_std.shift(1))
).fit().summary()


# %%
# Best and worst sectors over time
def sector_performance(models):
    perfs = []
    for date, model in models.items():
        perfs.append(
            model.res.posterior['1|sector'].median(
                dim=['chain', 'draw']
            ).to_dataframe().assign(date=model.date).reset_index()
        )
    df_p = pd.concat(perfs).rename(
        columns={'sector__factor_dim': 'sector', '1|sector': 'sharpe'}
    ).set_index(['sector', 'date']).unstack()
    return df_p


# %%
df_p = sector_performance(models_me)
# df_p = df_p.sort_values(df_p.columns[-1], ascending=False)
df_p = df_p.loc[(df_p > 0).mean(axis=1).sort_values(ascending=False).index, :]
# %%
df_p.columns = [d for _, d in df_p.columns]
# %%
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 18), height_ratios=[1, 10], width_ratios=[10, 1])
ax = axs[1][0]
c_ax = axs[1][1]
sns.heatmap(df_p, ax=ax, cmap='bwr', center=0, linewidth=0.1, cbar_ax=c_ax)
ax.set_title('Relative Sector Sharpe Ratio')
ax = axs[0][0]
plot_sharpe_dispersion(df_sr, ax=ax)
ax.set_xlim(df_sr.index[0], df_sr.index[-1])
axs[0][1].axis('off')
plt.tight_layout()
plt.show()
fig.savefig( plot_dir / 'sector_heatmap_monthly.png')