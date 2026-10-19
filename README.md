<div align="center">

# 🛒 E-Commerce Sales Analysis

### 🚀 Turning Raw Sales Data into Business Intelligence with Python & Power BI

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**An end-to-end analytics pipeline that cleans, analyzes, tests, and forecasts e-commerce sales data, then delivers the insights through an interactive Power BI dashboard.**

[Overview](#-overview) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Statistical Analysis & Forecasting](#-statistical-analysis--forecasting) • [Output](#-output-files) • [Contact](#-contact)

</div>

---

## 📖 Overview

> *"Raw e-commerce data is messy. It contains missing values, duplicate orders, inconsistent product names, and extreme outliers that hide the real story."*

This project solves that problem. It **automatically cleans** transactional sales data, **performs exploratory data analysis (EDA)**, **engineers business KPIs**, **tests business hypotheses**, **forecasts future revenue**, and **exports everything** into a Power BI dashboard for interactive reporting.

| 🎯 Goal | ✅ Result |
|:--------|:---------|
| Clean transactional data | No duplicates, no broken fields |
| Analyze sales performance | Revenue, profit, trends, categories |
| Understand customers | Segmentation by value and behavior |
| Test business questions | t-test, ANOVA and chi-square with effect sizes |
| Predict what's next | ARIMA / SARIMA forecast with 95% prediction intervals |
| Visualize insights | Python charts + Power BI dashboard |
| Save clean output | `cleaned_ecommerce_data.csv` ready for BI |

---

## ✨ Features

### 🟢 Python Analysis Pipeline
- 📅 Auto date parsing and time-series features
- 🧹 Duplicate removal and missing-value handling
- 💲 Numeric coercion for sales, profit, quantity
- 📊 KPI computation (revenue, profit, AOV, margin)
- 🚩 Outlier detection and flagging
- 📈 6+ EDA charts saved automatically
- 🧪 Hypothesis tests with p-values and effect sizes
- 🔮 Weekly revenue forecast with 95% prediction intervals

### 🔵 Power BI Dashboard
- 📌 KPI cards (Revenue, Orders, AOV, Profit)
- 📉 Time-series sales trends
- 🏆 Top products & categories
- 🌍 Regional performance breakdown
- 🎛️ Interactive slicers for custom exploration

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Abdalla820/E-Commerce-Sales.git
cd E-Commerce-Sales
```

### 2. Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

### 3. Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scipy *(hypothesis tests)*
- statsmodels *(ARIMA / SARIMA forecasting)*
- Power BI Desktop (free), to open the `.pbix` dashboard

---

## 🚀 Usage

### Basic Run

```bash
python E-Commerce.py
```

### Custom File Path

```bash
python E-Commerce.py path/to/your/data.csv
```

### Statistical Tests & Forecast

```bash
python analysis_simple.py
python analysis_simple.py path/to/your/data.csv path/to/output_folder
```

### Folder Structure

```text
📦 E-Commerce-Sales
 ┣ 📜 E-Commerce.py                  ← cleaning, EDA, KPIs, hypothesis tests and forecasting
 ┣ 📜 E-Commerce.pbix
 ┣ 📂 Dataset
 ┃ ┗ 📄 Ecommerce_Sales_Data_2024_2025.csv
 ┣ 📂 analysis_output
 ┃ ┣ 📄 cleaned_ecommerce_data.csv   ← generated
 ┃ ┣ 📄 phase1_summary.csv           ← generated
 ┃ ┣ 📄 phase2_metrics.csv           ← generated
 ┃ ┣ 📄 phase2_future_forecast.csv   ← generated
 ┃ ┗ 📄 *.png                        ← generated charts
 ┗ 📄 README.md                      ← this file
```

---

## 🧠 How It Works

### 📥 Step 1: Load & Inspect

- Reads the CSV/Excel file using pandas.
- Prints shape, dtypes, missing %, and duplicate count.
- Standardizes column names for consistency.

### 🧹 Step 2: Clean the Data

| Issue | Action |
|:------|:-------|
| Duplicate rows | Remove |
| Missing numeric values | Median imputation |
| Missing categorical values | Fill with "Unknown" |
| Invalid dates | Coerce to NaT |
| Non-numeric sales fields | Coerce to NaN → median |

### ⚙️ Step 3: Feature Engineering

- Extracts Year, Month, Quarter, Day of Week
- Computes Profit Margin and Unit Price
- Creates time-series groupings for trends

### 📊 Step 4: KPIs & Statistics

- Calculates: Total Revenue, Total Profit, Total Orders, AOV, Avg Margin
- Prints a plain-English summary of business performance

### 📈 Step 5: Visualize

Generates a set of charts :

1. Monthly sales trend
2. Top 10 products by revenue
3. Sales by category
4. Regional performance
5. Profit distribution
6. Correlation heatmap

### 💾 Step 6: Output

- Saves summary tables for Power BI
- Exports all charts as `.png`

---

## 🧪 Statistical Analysis & Forecasting

### Phase 2: Hypothesis Tests

| Test | Question | p-value | Effect size | Decision |
|:-----|:---------|:--------|:------------|:---------|
| Welch t-test | Order value: high (15-20%) vs low (0-10%) discount | 2.6e-12 | d = -0.20 (negligible) | Reject H0 |
| Welch t-test | Profit margin: high vs low discount | 0.88 | d ≈ 0 | Fail to reject H0 |
| One-way ANOVA | Order value across the 4 regions | 0.026 | η² = 0.0019 (negligible) | Reject H0 |
| Chi-square | Category × Payment Mode | 0.23 | V = 0.046 | Fail to reject H0 |
| Chi-square | Region × Payment Mode | 0.99 | V = 0.015 | Fail to reject H0 |
| Chi-square | Order-value tier × Payment Mode | 0.60 | V = 0.023 | Fail to reject H0 |

**Plain-English reading:** high-discount orders average about 15% less per order (96,752 vs 113,410), but profit margin is unchanged. Regional differences are statistically detectable but practically tiny, and payment mode is independent of category, region, and order size. Statistical significance alone is not business significance, which is why effect sizes are reported.

### Phase 3: Time-Series Forecast

| Step | Result |
|:-----|:-------|
| Series | Weekly revenue, 103 complete weeks |
| Stationarity | ADF and KPSS both indicate stationary, so d = 0 |
| Trend / seasonality | No trend (p = 0.94), no calendar seasonality (p = 0.98) |
| Split | 80% train / 20% test, chronological |
| Selected model | SARIMA(2,0,0)×(0,0,1,13), chosen by training AIC |
| Hold-out error | RMSE ≈ 1.25M, MAPE ≈ 20.3% |
| Interval quality | 95% prediction interval covered 90.5% of test weeks |
| 12-week outlook | About 5.1M per week, 95% range roughly 3.1M to 7.2M |

> ⚠️ **Honest note:** the sales series is flat and noisy. The model is well calibrated, but its accuracy is only marginally better than predicting the historical average, so the forecast is best used as a planning range rather than a precise prediction.

---

## 📊 Sample Output

```text
============================================================
  E-COMMERCE SALES ANALYSIS — START
============================================================
[INFO] Loading dataset: Dataset/Ecommerce_Sales_Data_2024_2025.csv
[INFO] Loaded shape: (5000, 14)
[INFO] Removed 0 duplicate rows

===== KEY PERFORMANCE INDICATORS =====
         total_revenue: 533,666,024.35
          total_profit: 79,708,734.91
          total_orders: 5,000
       avg_order_value: 106,733.20
     avg_profit_margin: 0.15

[INFO] Saved chart: analysis_output/01_monthly_sales_trend.png
[INFO] Saved chart: analysis_output/02_top_products.png
[INFO] Saved chart: analysis_output/03_sales_by_category.png
[INFO] Saved chart: analysis_output/04_sales_by_region.png
[INFO] Saved chart: analysis_output/05_profit_distribution.png
[INFO] Saved chart: analysis_output/06_correlation_heatmap.png
[INFO] Saved cleaned data: analysis_output/cleaned_ecommerce_data.csv
[INFO] Exported summary tables for Power BI.

============================================================
  PIPELINE COMPLETED SUCCESSFULLY ✅
============================================================
```

---

## 🎯 Key Design Decisions

| ✅ Principle | 💡 Why |
|:------------|:-------|
| Clean before analyzing | Garbage in → garbage out |
| Preserve real outliers | High-value orders are real, not errors |
| Median for numerics | Robust to skew in sales data |
| Separate BI exports | Power BI loads faster with summary tables |
| Report effect sizes with p-values | Large samples make tiny differences "significant" |
| Pick the model by training AIC | Choosing by test error would leak the hold-out set |
| Modular functions | Each step is reusable and testable |

---

## 📁 Output Files

| 📄 File | 📝 Description |
|:--------|:---------------|
| `cleaned_ecommerce_data.csv` | Cleaned dataset ready for analysis |
| `phase1_summary.csv` | Hypothesis test results (p-values, effect sizes, decisions) |
| `phase2_metrics.csv` | Forecast model comparison (RMSE, MAPE, interval coverage) |
| `phase2_future_forecast.csv` | 12-week forecast with 95% bounds |
| `*.png` | All EDA and forecast visualizations |
| `E-Commerce.pbix` | Interactive Power BI dashboard |

---

## 🧰 Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=python&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)

</div>

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. 🍴 Fork the repository
2. 🌿 Create a feature branch
3. 💾 Commit your changes
4. 🚀 Push and open a Pull Request

---

## 📜 License

This project is licensed under the MIT License, free to use and modify.

---

## 📧 Contact

<div align="center">

**Abdalla Mohamed Farouk**

[![Email](https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white)](mailto:abdallamohamed8354@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdalla-mohamed-farouk)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white)](https://github.com/Abdalla820)

</div>

<div align="center">

⭐ If you found this project useful, please give it a star!

</div>
