import itertools
import os
import sys
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "Dataset", "Ecommerce_Sales_Data_2024_2025.csv")
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
if not os.path.exists(CSV_PATH):
    alt_csv = os.path.join(SCRIPT_DIR, "Ecommerce_Sales_Data_2024_2025.csv")
    if os.path.exists(alt_csv):
        CSV_PATH = alt_csv
    else:
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}\nPlace it in the project root or in the Dataset folder.")

OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(SCRIPT_DIR, "analysis_output")
ALPHA = 0.05           
TEST_SHARE = 0.20      
FUTURE_WEEKS = 12      
SEASONS = (4, 13)      
os.makedirs(OUT_DIR, exist_ok=True)


def title(text):
    print(f"\n{'=' * 80}\n{text}\n{'=' * 80}")

def decision(p):
    return "REJECT H0" if p < ALPHA else "fail to reject H0"

def size_label(x, cutoffs, names):
    """Turn an effect size into a word, e.g. 0.12 -> 'negligible'."""
    for cutoff, name in zip(cutoffs, names):
        if abs(x) < cutoff:
            return name
    return names[-1]

def cohens_d(a, b):
    pooled_sd = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled_sd

def d_label(d):
    return size_label(d, (0.2, 0.5, 0.8), ("negligible", "small", "medium", "large"))

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}", dpi=150)
    plt.close(fig)


def load_data():
    df = pd.read_csv(CSV_PATH, parse_dates=["Order Date"])
    df = df.drop_duplicates("Order ID").dropna()
    df["Profit Margin"] = df["Profit"] / df["Sales"]
    df["Discount Group"] = np.where(df["Discount"] >= 15, "High (15-20%)", "Low (0-10%)")
    df["Order Value Tier"] = np.where(df["Sales"] >= df["Sales"].median(), "High", "Low")

    title("DATA OVERVIEW")
    print(f"{len(df):,} orders from {df['Order Date'].min().date()} to {df['Order Date'].max().date()}")
    return df


def t_test(df, metric):
    """Welch t-test: high-discount orders (B) vs low-discount orders (A)."""
    a = df.loc[df["Discount"] < 15, metric]
    b = df.loc[df["Discount"] >= 15, metric]
    t, p = stats.ttest_ind(b, a, equal_var=False)
    d = cohens_d(b, a)
    print(f"\n[{metric}]")
    print(f"  H0: mean {metric} is the same for high- and low-discount orders")
    print(f"  H1: mean {metric} differs between high- and low-discount orders")
    print(f"  mean low = {a.mean():,.2f} | mean high = {b.mean():,.2f}")
    print(f"  t = {t:.2f}, p = {p:.2e} -> {decision(p)}")
    print(f"  Cohen's d = {d:.3f} ({d_label(d)})")
    return p, d

def anova_by_region(df):
    """One-way ANOVA on order value across regions, with Tukey post-hoc."""
    groups = [g["Sales"] for _, g in df.groupby("Region")]
    f, p = stats.f_oneway(*groups)

    grand_mean = df["Sales"].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    eta2 = ss_between / ((df["Sales"] - grand_mean) ** 2).sum()

    print("\n[Sales by Region]  H0: all region means equal   H1: at least one differs")
    print(df.groupby("Region")["Sales"].mean().round(0).to_string())
    print(f"  F = {f:.2f}, p = {p:.4f} -> {decision(p)}")
    print(f"  eta^2 = {eta2:.4f} ({size_label(eta2, (0.01, 0.06, 0.14), ('negligible', 'small', 'medium', 'large'))})")
    if p < ALPHA:
        print("\n  Tukey post-hoc (which pairs differ?):")
        print(pairwise_tukeyhsd(df["Sales"], df["Region"], alpha=ALPHA).summary())
    return p, eta2

def chi_square(df, col1, col2):
    """Chi-square test of independence between two categorical columns."""
    table = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, _ = stats.chi2_contingency(table)
    v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))) 
    print(f"\n[{col1} x {col2}]  H0: independent   H1: associated")
    print(f"  chi2({dof}) = {chi2:.2f}, p = {p:.4f} -> {decision(p)}")
    print(f"  Cramer's V = {v:.3f} ({size_label(v, (0.1, 0.3, 0.5), ('negligible', 'small', 'medium', 'large'))})")
    return p, v

def run_phase1(df):
    title(f"PHASE 1 - HYPOTHESIS TESTS (alpha = {ALPHA})")
    rows = []

    print("\n--- Test 1: t-test, high vs low discount ---")
    p, d = t_test(df, "Sales")
    rows.append(("t-test", "Order value: high vs low discount", p, f"d = {d:.2f}"))
    p, d = t_test(df, "Profit Margin")
    rows.append(("t-test", "Profit margin: high vs low discount", p, f"d = {d:.2f}"))

    print("\n--- Test 2: one-way ANOVA ---")
    p, eta2 = anova_by_region(df)
    rows.append(("ANOVA", "Order value across regions", p, f"eta2 = {eta2:.4f}"))

    print("\n--- Test 3: chi-square ---")
    for col1, col2 in [("Category", "Payment Mode"), ("Region", "Payment Mode"),
                       ("Order Value Tier", "Payment Mode")]:
        p, v = chi_square(df, col1, col2)
        rows.append(("Chi-square", f"{col1} x {col2}", p, f"V = {v:.3f}"))

    summary = pd.DataFrame(rows, columns=["Test", "Question", "p-value", "Effect size"])
    summary["Decision"] = summary["p-value"].apply(decision)
    title("PHASE 1 SUMMARY")
    print(summary.to_string(index=False))
    summary.to_csv(f"{OUT_DIR}/phase1_summary.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    sns.boxplot(data=df, x="Discount Group", y="Sales", showfliers=False, ax=ax[0])
    ax[0].set_title("Order value by discount group")
    sns.barplot(data=df, x="Region", y="Sales", errorbar=("ci", 95), capsize=0.1, ax=ax[1])
    ax[1].set_title("Mean order value by region (95% CI)")
    sns.heatmap(pd.crosstab(df["Category"], df["Payment Mode"], normalize="index"),
                annot=True, fmt=".2f", cmap="Blues", ax=ax[2])
    ax[2].set_title("Payment mix by category")
    save(fig, "phase1_tests.png")


def make_weekly_series(df):
    """Weekly revenue, dropping the partial first and last weeks."""
    ts = df.set_index("Order Date")["Sales"].resample("W-SUN").sum().iloc[1:-1]
    ts.index.freq = "W-SUN"
    print(f"{len(ts)} weekly points: {ts.index.min().date()} -> {ts.index.max().date()}")
    return ts


def choose_differencing(ts):
    """Difference only if ADF can't reject a unit root AND KPSS rejects stationarity."""
    d, series = 0, ts
    while d < 2:
        p_adf = adfuller(series, autolag="BIC")[1]     
        p_kpss = kpss(series, regression="c", nlags="auto")[1]  
        print(f"  d={d}: ADF p = {p_adf:.4f}, KPSS p = {p_kpss:.3f}")
        if p_adf < ALPHA or p_kpss >= ALPHA:
            print(f"  -> stationary, using d = {d}")
            return d, series
        d += 1
        series = series.diff().dropna()
    return d, series

def plot_eda(ts, stationary_series):
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    ts.plot(ax=ax[0, 0], label="Weekly sales")
    ts.rolling(8).mean().plot(ax=ax[0, 0], color="red", label="8-week mean")
    ax[0, 0].set_title("Weekly revenue (look for trend)")
    ax[0, 0].legend()
    ts.resample("MS").sum().div(1e6).plot(ax=ax[0, 1], marker="o")
    ax[0, 1].set_title("Monthly revenue, millions (look for seasonality)")
    plot_acf(stationary_series, lags=30, ax=ax[1, 0])
    plot_pacf(stationary_series, lags=30, ax=ax[1, 1], method="ywm")
    save(fig, "phase2_eda.png")

    slope, _, _, p_trend, _ = stats.linregress(np.arange(len(ts)), ts.values)
    print(f"  Linear trend: {slope:,.0f} per week (p = {p_trend:.3f})")

  
    by_month = [g.values for _, g in ts.groupby(ts.index.month)]
    p_season = stats.f_oneway(*by_month)[1]
    print(f"  Month-of-year seasonality: p = {p_season:.3f} "
          f"({'seasonal pattern present' if p_season < ALPHA else 'no calendar seasonality'})")


def fit_sarima(series, order, seasonal_order=(0, 0, 0, 0)):
    return SARIMAX(series, order=order, seasonal_order=seasonal_order,
                   trend="c" if order[1] == 0 else "n").fit(disp=False)


def grid_search(train, d):
    """Try ARIMA and SARIMA settings; return the best of each by AIC (train only)."""
    candidates = [("ARIMA", (p, d, q), (0, 0, 0, 0))
                  for p, q in itertools.product(range(4), range(4))]
    candidates += [("SARIMA", (p, d, q), (P, 0, Q, m))
                   for m in SEASONS
                   for p, q, P, Q in itertools.product(range(3), range(3), range(2), range(2))
                   if P or Q]

    rows = []
    for kind, order, seasonal in candidates:
        try:
            rows.append((kind, order, seasonal, fit_sarima(train, order, seasonal).aic))
        except Exception:
            continue
    table = pd.DataFrame(rows, columns=["kind", "order", "seasonal", "aic"]).sort_values("aic")
    print(table.head(5).to_string(index=False))
    return {kind: table[table["kind"] == kind].iloc[0] for kind in ("ARIMA", "SARIMA")}


def evaluate(train, test, best):
    """Forecast the test period with each model and compare against simple benchmarks."""
    rmse = lambda y, yhat: float(np.sqrt(np.mean((y - yhat) ** 2)))
    mape = lambda y, yhat: float(np.mean(np.abs((y - yhat) / y)) * 100)

    rows, forecasts = [], {}
    for kind, row in best.items():
        name = f"{kind}{row['order']}" + (f"x{row['seasonal']}" if kind == "SARIMA" else "")
        fc = fit_sarima(train, row["order"], row["seasonal"]).get_forecast(len(test))
        mean, ci = fc.predicted_mean, fc.conf_int(alpha=ALPHA)
        inside = ((test.values >= ci.iloc[:, 0].values) & (test.values <= ci.iloc[:, 1].values)).mean()
        rows.append((name, rmse(test.values, mean.values), mape(test.values, mean.values),
                     inside * 100, row["aic"]))
        forecasts[name] = (row, mean, ci)

    for name, value in [("Benchmark: train mean", train.mean()), ("Benchmark: last value", train.iloc[-1])]:
        rows.append((name, rmse(test.values, value), mape(test.values, value), np.nan, np.nan))

    metrics = pd.DataFrame(rows, columns=["Model", "RMSE", "MAPE %", "PI coverage %", "Train AIC"])
    print(metrics.round(2).to_string(index=False))
    metrics.to_csv(f"{OUT_DIR}/phase2_metrics.csv", index=False)

    champion = metrics.iloc[:2].sort_values("Train AIC").iloc[0]["Model"]
    print(f"\nChampion (lowest train AIC): {champion}")
    return champion, forecasts[champion]


def plot_forecast(history, forecast_mean, ci, name, filename, actual=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(history.index, history, label="History / train", color="#1f77b4")
    if actual is not None:
        ax.plot(actual.index, actual, label="Actual (test)", color="black")
    ax.plot(forecast_mean.index, forecast_mean, label=f"Forecast {name}", color="#d62728")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#ebe40d", alpha=0.2,
                    label="95% prediction interval")
    ax.legend()
    save(fig, filename)


def run_phase2(df):
    title("PHASE 2 - FORECASTING")
    ts = make_weekly_series(df)

    print("\nStep 1: stationarity")
    d, stationary_series = choose_differencing(ts)

    print("\nStep 2: exploratory plots")
    plot_eda(ts, stationary_series)

    split = int(len(ts) * (1 - TEST_SHARE))
    train, test = ts.iloc[:split], ts.iloc[split:]
    print(f"\nStep 3: split -> train {len(train)} weeks, test {len(test)} weeks")

    print("\nStep 4: parameter search (top 5 by AIC)")
    best = grid_search(train, d)

    print("\nStep 5: hold-out evaluation")
    champion, (row, mean, ci) = evaluate(train, test, best)
    plot_forecast(train, mean, ci, champion, "phase2_forecast_holdout.png", actual=test)

    print(f"\nStep 6: refit on all data, forecast next {FUTURE_WEEKS} weeks")
    final = fit_sarima(ts, row["order"], row["seasonal"]).get_forecast(FUTURE_WEEKS)
    future_mean, future_ci = final.predicted_mean, final.conf_int(alpha=ALPHA)
    future = pd.DataFrame({"forecast": future_mean, "lower_95": future_ci.iloc[:, 0],
                           "upper_95": future_ci.iloc[:, 1]})
    print(future.round(0).to_string())
    future.to_csv(f"{OUT_DIR}/phase2_future_forecast.csv")
    plot_forecast(ts, future_mean, future_ci, champion, "phase2_forecast_future.png")


if __name__ == "__main__":
    data = load_data()
    run_phase1(data)
    run_phase2(data)
    title(f"DONE - results saved in {OUT_DIR}")