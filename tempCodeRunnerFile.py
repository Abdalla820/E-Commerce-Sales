"""
E-commerce Sales 2024-2025
Phase 2: Statistical Inferential Hypothesis Analysis  ("The Why")
Phase 3: Predictive Business Forecaster (ARIMA / SARIMA) ("What's Next")

Usage:
    pip install pandas numpy scipy statsmodels matplotlib seaborn
    python phase2_phase3_analysis.py [path_to_csv] [output_dir]
"""
import sys
import os
import warnings
import itertools

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "Ecommerce_Sales_Data_2024_2025.csv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "outputs_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA = 0.05


def banner(text):
    print("\n" + "=" * 90)
    print(text)
    print("=" * 90)


def verdict(p, alpha=ALPHA):
    return "REJECT H0 (statistically significant)" if p < alpha else "FAIL TO REJECT H0 (not significant)"


def cohens_d(a, b):
    """Cohen's d using pooled standard deviation."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled


def interpret_d(d):
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def welch_ci(a, b, conf=0.95):
    """Confidence interval for mean(a) - mean(b), Welch."""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dfree = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(1 - (1 - conf) / 2, dfree)
    diff = a.mean() - b.mean()
    return diff - t * se, diff + t * se


# ----------------------------------------------------------------------------------
# 0. LOAD & CLEAN
# ----------------------------------------------------------------------------------
df = pd.read_csv(CSV_PATH, parse_dates=["Order Date"])
df.columns = [c.strip() for c in df.columns]
df = df.drop_duplicates(subset="Order ID").dropna()
df["Profit Margin"] = df["Profit"] / df["Sales"]
df["Discount Group"] = np.where(df["Discount"] >= 15, "High discount (15-20%)", "Low discount (0-10%)")

banner("DATA OVERVIEW")
print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
print(f"Date range: {df['Order Date'].min().date()} -> {df['Order Date'].max().date()}")
print(df[["Quantity", "Unit Price", "Discount", "Sales", "Profit", "Profit Margin"]].describe().round(2))

# ==================================================================================
# PHASE 2 - HYPOTHESIS TESTING
# ==================================================================================
banner("PHASE 2 - STATISTICAL INFERENTIAL HYPOTHESIS ANALYSIS")
print(f"Significance level for ALL tests: alpha = {ALPHA}")

# ----------------------------------------------------------------------------------
# TEST 1 - Independent two-sample (Welch) T-TEST: discount strategy
# ----------------------------------------------------------------------------------
banner("TEST 1 - A/B STYLE T-TEST: High-discount (Group B) vs Low-discount (Group A)")

A = df.loc[df["Discount"] < 15]
B = df.loc[df["Discount"] >= 15]

# 1a. Average order value
print("\n--- 1a. Metric: Average Order Value (Sales per order) ---")
print("H0: mu_AOV(high discount) = mu_AOV(low discount)")
print("H1: mu_AOV(high discount) != mu_AOV(low discount)   (two-sided)")
a, b = A["Sales"], B["Sales"]
lev = stats.levene(a, b)
print(f"Levene test for equal variances: p = {lev.pvalue:.4f} -> "
      f"{'equal variances' if lev.pvalue >= ALPHA else 'unequal variances -> use Welch'}")
t, p = stats.ttest_ind(b, a, equal_var=False)
p_t1 = p
d = cohens_d(b, a)
lo, hi = welch_ci(b, a)
u = stats.mannwhitneyu(b, a, alternative="two-sided")
print(f"n(A)={len(a):,}  mean(A)={a.mean():,.0f}   n(B)={len(b):,}  mean(B)={b.mean():,.0f}")
print(f"Welch t = {t:.3f}, p = {p:.3e}  -> {verdict(p)}")
print(f"Mean difference (B-A) = {b.mean() - a.mean():,.0f}  95% CI [{lo:,.0f}, {hi:,.0f}]")
print(f"Cohen's d = {d:.3f} ({interpret_d(d)} effect)")
print(f"Robustness (Mann-Whitney U, non-parametric): p = {u.pvalue:.3e}")

# 1b. Profit margin
print("\n--- 1b. Metric: Profit Margin (Profit / Sales) ---")
print("H0: mu_margin(high discount) = mu_margin(low discount)")
print("H1: mu_margin(high discount) != mu_margin(low discount)")
a2, b2 = A["Profit Margin"], B["Profit Margin"]
t2, p2 = stats.ttest_ind(b2, a2, equal_var=False)
d2 = cohens_d(b2, a2)
lo2, hi2 = welch_ci(b2, a2)
print(f"mean(A)={a2.mean():.4f}   mean(B)={b2.mean():.4f}")
print(f"Welch t = {t2:.3f}, p = {p2:.4f}  -> {verdict(p2)}")
print(f"Mean difference (B-A) = {b2.mean() - a2.mean():.4f}  95% CI [{lo2:.4f}, {hi2:.4f}]")
print(f"Cohen's d = {d2:.3f} ({interpret_d(d2)} effect)")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sns.boxplot(data=df, x="Discount Group", y="Sales", ax=ax[0], showfliers=False,
            order=["Low discount (0-10%)", "High discount (15-20%)"])
ax[0].set_title(f"Order value by discount group\nWelch p={p:.2e}, Cohen's d={d:.2f}")
sns.boxplot(data=df, x="Discount Group", y="Profit Margin", ax=ax[1], showfliers=False,
            order=["Low discount (0-10%)", "High discount (15-20%)"])
ax[1].set_title(f"Profit margin by discount group\nWelch p={p2:.3f}, Cohen's d={d2:.2f}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/test1_ttest_discount.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------------
# TEST 2 - ONE-WAY ANOVA: sales across regions
# ----------------------------------------------------------------------------------
banner("TEST 2 - ONE-WAY ANOVA: Average order value across the 4 regions")
print("H0: mu_North = mu_South = mu_East = mu_West")
print("H1: at least one region mean differs")

groups = [g["Sales"].values for _, g in df.groupby("Region")]
labels = sorted(df["Region"].unique())
lev = stats.levene(*groups)
F, p = stats.f_oneway(*groups)
p_anova = p
H, p_kw = stats.kruskal(*groups)

grand = df["Sales"].mean()
ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_total = ((df["Sales"] - grand) ** 2).sum()
eta2 = ss_between / ss_total
k, N = len(groups), len(df)
omega2 = (ss_between - (k - 1) * ((ss_total - ss_between) / (N - k))) / (ss_total + (ss_total - ss_between) / (N - k))

print(df.groupby("Region")["Sales"].agg(["count", "mean", "std"]).round(0))
print(f"\nLevene (homogeneity of variance): p = {lev.pvalue:.4f}")
print(f"ANOVA F({k - 1},{N - k}) = {F:.3f}, p = {p:.4f}  -> {verdict(p)}")
print(f"Effect size: eta^2 = {eta2:.4f}, omega^2 = {omega2:.4f} "
      f"({'negligible' if eta2 < .01 else 'small' if eta2 < .06 else 'medium' if eta2 < .14 else 'large'}; "
      f"region explains {eta2 * 100:.2f}% of order-value variance)")
print(f"Robustness (Kruskal-Wallis): H = {H:.3f}, p = {p_kw:.4f}")

if p < ALPHA:
    print("\nPost-hoc Tukey HSD (which regions differ?):")
    tuk = pairwise_tukeyhsd(df["Sales"], df["Region"], alpha=ALPHA)
    print(tuk.summary())
    tk = pd.DataFrame(tuk._results_table.data[1:], columns=tuk._results_table.data[0])
    for _, r in tk.iterrows():
        pair_a = df.loc[df["Region"] == r["group1"], "Sales"]
        pair_b = df.loc[df["Region"] == r["group2"], "Sales"]
        print(f"  {r['group1']} vs {r['group2']}: Cohen's d = {cohens_d(pair_b, pair_a):.3f} "
              f"({interpret_d(cohens_d(pair_b, pair_a))})")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(data=df, x="Region", y="Sales", order=labels, errorbar=("ci", 95), ax=ax, capsize=.1)
ax.set_title(f"Mean order value by region (95% CI)\nANOVA p={p:.4f}, eta^2={eta2:.4f}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/test2_anova_region.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------------
# TEST 3 - CHI-SQUARE: independence of categorical variables
# ----------------------------------------------------------------------------------
banner("TEST 3 - CHI-SQUARE TEST OF INDEPENDENCE")


def chi_square(col1, col2, title):
    print(f"\n--- {title} ---")
    print(f"H0: {col1} and {col2} are independent")
    print(f"H1: {col1} and {col2} are associated")
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    print(f"chi2({dof}) = {chi2:.3f}, p = {p:.4f}  -> {verdict(p)}")
    print(f"Cramer's V = {v:.4f} ({'negligible' if v < .1 else 'small' if v < .3 else 'medium' if v < .5 else 'large'} association)")
    print(f"Min expected cell count = {exp.min():.1f} (rule of thumb: >= 5 -> {'OK' if exp.min() >= 5 else 'WARNING'})")
    return ct, p, v


ct1, p_c1, v1 = chi_square("Category", "Payment Mode", "3a. Product Category vs Payment Mode")
ct2, p_c2, v2 = chi_square("Region", "Payment Mode", "3b. Region vs Payment Mode")

median_sales = df["Sales"].median()
df["Order Value Tier"] = np.where(df["Sales"] >= median_sales, "High-value", "Low-value")
ct3, p_c3, v3 = chi_square("Order Value Tier", "Payment Mode", "3c. Order Value Tier (above/below median) vs Payment Mode")

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.heatmap(pd.crosstab(df["Category"], df["Payment Mode"], normalize="index"), annot=True, fmt=".2f",
            cmap="Blues", ax=ax[0])
ax[0].set_title(f"Payment mix by category (row %)\nchi2 p={p_c1:.3f}, V={v1:.3f}")
sns.heatmap(pd.crosstab(df["Region"], df["Payment Mode"], normalize="index"), annot=True, fmt=".2f",
            cmap="Blues", ax=ax[1])
ax[1].set_title(f"Payment mix by region (row %)\nchi2 p={p_c2:.3f}, V={v2:.3f}")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/test3_chi_square.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------------------
# PHASE 2 SUMMARY
# ----------------------------------------------------------------------------------
banner("PHASE 2 - SUMMARY TABLE")
summary = pd.DataFrame([
    ["Welch t-test", "AOV: high vs low discount", p_t1, f"d={d:.2f} ({interpret_d(d)})", verdict(p_t1)],
    ["Welch t-test", "Profit margin: high vs low discount", p2, f"d={d2:.2f} ({interpret_d(d2)})", verdict(p2)],
    ["One-way ANOVA", "AOV across 4 regions", p_anova, f"eta2={eta2:.4f}", verdict(p_anova)],
    ["Chi-square", "Category x Payment Mode", p_c1, f"V={v1:.3f}", verdict(p_c1)],
    ["Chi-square", "Region x Payment Mode", p_c2, f"V={v2:.3f}", verdict(p_c2)],
    ["Chi-square", "Order value tier x Payment Mode", p_c3, f"V={v3:.3f}", verdict(p_c3)],
], columns=["Test", "Question", "p-value", "Effect size", "Decision (alpha=0.05)"])
print(summary.to_string(index=False))
summary.to_csv(f"{OUT_DIR}/phase2_summary.csv", index=False)

# ==================================================================================
# PHASE 3 - TIME-SERIES FORECASTING (ARIMA / SARIMA)
# ==================================================================================
banner("PHASE 3 - PREDICTIVE BUSINESS FORECASTER (Time-Series, Path A)")

# --- 3.0 Build a clean weekly series (drop incomplete first/last weeks) --------------
ts = df.set_index("Order Date")["Sales"].resample("W-SUN").sum()
ts = ts.iloc[1:-1]            # first and last bins are partial weeks -> would bias the model
ts.index.freq = "W-SUN"
print(f"Weekly revenue series: {len(ts)} weeks ({ts.index.min().date()} -> {ts.index.max().date()})")

# --- 3.1 STATIONARITY (ADF) -----------------------------------------------------------
banner("Step 3.1 - Stationarity check (Augmented Dickey-Fuller)")


def adf_report(series, name):
    # BIC lag selection is more parsimonious than AIC and avoids the huge-lag
    # over-fitting that can wrongly make a flat, noisy series look non-stationary.
    res = adfuller(series.dropna(), autolag="BIC")
    print(f"{name}: ADF = {res[0]:.3f}, p = {res[1]:.4f}, lags = {res[2]}  -> "
          f"{'STATIONARY' if res[1] < ALPHA else 'NON-STATIONARY'}")
    return res[1]


def kpss_report(series, name):
    # KPSS has the opposite null (H0: stationary), so it is a useful cross-check.
    stat, p_k, _, _ = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"{name}: KPSS = {stat:.3f}, p = {p_k:.3f}  (H0 = stationary) -> "
          f"{'STATIONARY' if p_k >= ALPHA else 'NON-STATIONARY'}")
    return p_k


print("ADF   H0: series has a unit root (non-stationary)   H1: series is stationary")
print("KPSS  H0: series is stationary                      H1: series has a unit root")
p_adf = adf_report(ts, "Level series")
p_kpss = kpss_report(ts, "Level series")

print("\nSensitivity of ADF to the number of lags (robustness check):")
for ml in (0, 2, 4, 8, 12):
    r = adfuller(ts, maxlag=ml, autolag=None)
    print(f"  lags={ml:>2}: ADF = {r[0]:6.2f}, p = {r[1]:.4f}")

d_order = 0
work = ts.copy()
# Difference only when ADF cannot reject the unit root AND KPSS rejects stationarity.
while (p_adf >= ALPHA and p_kpss < ALPHA) and d_order < 2:
    d_order += 1
    work = work.diff().dropna()
    p_adf = adf_report(work, f"Differenced d={d_order}")
    p_kpss = kpss_report(work, f"Differenced d={d_order}")
print(f"=> differencing order selected: d = {d_order}")

# --- 3.2 EDA ------------------------------------------------------------------------
banner("Step 3.2 - EDA: trend, seasonality, ACF / PACF")
fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ts.plot(ax=ax[0], label="Weekly sales")
ts.rolling(8).mean().plot(ax=ax[0], label="8-week rolling mean", color="red")
ax[0].set_title("Weekly revenue")
ax[0].legend()
monthly = df.set_index("Order Date")["Sales"].resample("MS").sum().iloc[:-1]
(monthly / 1e6).plot(ax=ax[1], marker="o")
ax[1].set_title("Monthly revenue (millions, incomplete final month excluded)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_time_series.png", dpi=150)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(work, lags=30, ax=ax[0])
plot_pacf(work, lags=30, ax=ax[1], method="ywm")
ax[0].set_title(f"ACF (series used for ARIMA, d={d_order})")
ax[1].set_title("PACF")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_acf_pacf.png", dpi=150)
plt.close()

dec = seasonal_decompose(ts, model="additive", period=26)  # 103 weeks < 2 full years, so use a half-year cycle
dec.plot().set_size_inches(12, 8)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_decomposition.png", dpi=150)
plt.close()

# Seasonality / trend evidence
slope, intercept, r, p_trend, _ = stats.linregress(np.arange(len(ts)), ts.values)
print(f"Linear trend: slope = {slope:,.0f} per week, p = {p_trend:.4f}")
monthly_groups = [g.values for _, g in ts.groupby(ts.index.month)]
Fm, p_m = stats.f_oneway(*monthly_groups)
print(f"Month-of-year seasonality (ANOVA on weekly sales by calendar month): F = {Fm:.2f}, p = {p_m:.4f}")
lb = acorr_ljungbox(ts, lags=[4, 13, 26], return_df=True)
print("Ljung-Box (is there any autocorrelation to model?):")
print(lb.round(4))

# --- 3.3 TRAIN / TEST SPLIT -----------------------------------------------------------
banner("Step 3.3 - Train / test split (80 / 20, chronological)")
split = int(len(ts) * 0.8)
train, test = ts.iloc[:split], ts.iloc[split:]
print(f"Train: {len(train)} weeks ({train.index.min().date()} -> {train.index.max().date()})")
print(f"Test : {len(test)} weeks ({test.index.min().date()} -> {test.index.max().date()})")

# --- 3.4 MODEL SELECTION (grid search on AIC, TRAIN ONLY) ------------------------------
banner("Step 3.4 - ARIMA / SARIMA parameter search (minimise AIC on training data)")
results = []
# Non-seasonal ARIMA
for p_, q_ in itertools.product(range(0, 4), range(0, 4)):
    try:
        m = SARIMAX(train, order=(p_, d_order, q_), trend="c" if d_order == 0 else "n",
                    enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
        results.append(("ARIMA", (p_, d_order, q_), None, m.aic, m.bic))
    except Exception:
        pass
# Seasonal candidates: m=4 (~monthly cycle in weeks) and m=13 (quarterly cycle in weeks)
for season in (4, 13):
    for p_, q_, P_, Q_ in itertools.product(range(0, 3), range(0, 3), range(0, 2), range(0, 2)):
        if P_ == 0 and Q_ == 0:
            continue
        try:
            m = SARIMAX(train, order=(p_, d_order, q_), seasonal_order=(P_, 0, Q_, season),
                        trend="c" if d_order == 0 else "n",
                        enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
            results.append(("SARIMA", (p_, d_order, q_), (P_, 0, Q_, season), m.aic, m.bic))
        except Exception:
            pass

res_df = pd.DataFrame(results, columns=["Type", "order", "seasonal_order", "AIC", "BIC"]).sort_values("AIC")
print(res_df.head(8).to_string(index=False))

best_arima = res_df[res_df["Type"] == "ARIMA"].iloc[0]
best_sarima = res_df[res_df["Type"] == "SARIMA"].iloc[0]
print(f"\nBest ARIMA : {best_arima['order']}  AIC={best_arima['AIC']:.1f}")
print(f"Best SARIMA: {best_sarima['order']}x{best_sarima['seasonal_order']}  AIC={best_sarima['AIC']:.1f}")

# --- 3.5 EVALUATE ON HOLD-OUT with 95% PREDICTION INTERVALS ---------------------------
banner("Step 3.5 - Hold-out evaluation with 95% prediction intervals")


def fit_model(series, order, seasonal_order=None):
    return SARIMAX(series, order=order, seasonal_order=seasonal_order or (0, 0, 0, 0),
                   trend="c" if order[1] == 0 else "n",
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat):
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


candidates = {
    f"ARIMA{best_arima['order']}": (best_arima["order"], None),
    f"SARIMA{best_sarima['order']}x{best_sarima['seasonal_order']}": (best_sarima["order"], best_sarima["seasonal_order"]),
}
metrics, fcs = [], {}
for name, (order, sorder) in candidates.items():
    fit = fit_model(train, order, sorder)
    fc = fit.get_forecast(steps=len(test))
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=ALPHA)
    cover = float(((test.values >= ci.iloc[:, 0].values) & (test.values <= ci.iloc[:, 1].values)).mean() * 100)
    metrics.append([name, rmse(test.values, mean.values), mape(test.values, mean.values), cover, fit.aic])
    fcs[name] = (fit, mean, ci)

# naive benchmarks: train mean and last-value
for name, yhat in [("Benchmark: train mean", np.full(len(test), train.mean())),
                   ("Benchmark: last value (naive)", np.full(len(test), train.iloc[-1]))]:
    metrics.append([name, rmse(test.values, yhat), mape(test.values, yhat), np.nan, np.nan])

met = pd.DataFrame(metrics, columns=["Model", "RMSE", "MAPE %", "95% PI coverage %", "AIC (train)"])
print(met.round(2).to_string(index=False))
met.to_csv(f"{OUT_DIR}/phase3_model_metrics.csv", index=False)

# Champion chosen by TRAINING AIC only (choosing by test error would leak the hold-out set).
# Both candidates share d, so their AICs are computed on the same observations and are comparable.
champ_name = met.iloc[:2].sort_values("AIC (train)").iloc[0]["Model"]
champ_fit, champ_mean, champ_ci = fcs[champ_name]
print(f"\nChampion model (lowest training AIC): {champ_name}")

resid = champ_fit.resid.iloc[max(d_order, 1):]
lbr = acorr_ljungbox(resid, lags=[10], return_df=True)
print(f"Residual Ljung-Box (lag 10): p = {lbr['lb_pvalue'].iloc[0]:.4f} -> "
      f"{'residuals look like white noise (good)' if lbr['lb_pvalue'].iloc[0] > ALPHA else 'residual autocorrelation remains'}")
print(f"Residual normality (Shapiro-Wilk): p = {stats.shapiro(resid).pvalue:.4f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(train.index, train, label="Train", color="#1f77b4")
ax.plot(test.index, test, label="Actual (test)", color="black")
ax.plot(champ_mean.index, champ_mean, label=f"Forecast: {champ_name}", color="#d62728")
ax.fill_between(champ_ci.index, champ_ci.iloc[:, 0], champ_ci.iloc[:, 1], color="#d62728", alpha=.2,
                label="95% prediction interval")
ax.set_title("Hold-out forecast with 95% prediction interval")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/forecast_holdout.png", dpi=150)
plt.close()

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
resid.plot(ax=ax[0, 0], title="Residuals")
sns.histplot(resid, kde=True, ax=ax[0, 1]).set_title("Residual distribution")
plot_acf(resid, lags=25, ax=ax[1, 0])
stats.probplot(resid, plot=ax[1, 1])
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/residual_diagnostics.png", dpi=150)
plt.close()

# --- 3.6 REFIT ON ALL DATA -> TRUE FUTURE FORECAST (next 12 weeks) ---------------------
banner("Step 3.6 - Future forecast (next 12 weeks), refit on full data")
order, sorder = candidates[champ_name]
final = fit_model(ts, order, sorder)
H_FUT = 12
ff = final.get_forecast(steps=H_FUT)
fmean, fci = ff.predicted_mean, ff.conf_int(alpha=ALPHA)
future = pd.DataFrame({"forecast": fmean, "lower_95": fci.iloc[:, 0], "upper_95": fci.iloc[:, 1]})
print(future.round(0).to_string())
future.to_csv(f"{OUT_DIR}/future_forecast_12w.csv")
print(f"\nExpected revenue over next {H_FUT} weeks: {future['forecast'].sum():,.0f}")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts.index, ts, label="History", color="#1f77b4")
ax.plot(fmean.index, fmean, color="#d62728", label="Forecast")
ax.fill_between(fci.index, fci.iloc[:, 0], fci.iloc[:, 1], color="#d62728", alpha=.2, label="95% prediction interval")
ax.set_title(f"{H_FUT}-week revenue forecast with 95% prediction interval")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/forecast_future.png", dpi=150)
plt.close()

banner(f"DONE - tables and figures saved in ./{OUT_DIR}/")