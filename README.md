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

**An end-to-end analytics pipeline that cleans, analyzes, and visualizes e-commerce sales data — then delivers insights through an interactive Power BI dashboard.**

[Overview](#-overview) • [Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Output](#-output) • [Contact](#-contact)

</div>

---

## 📖 Overview

> *"Raw e-commerce data is messy. It contains missing values, duplicate orders, inconsistent product names, and extreme outliers that hide the real story."*

This project solves that problem. It **automatically cleans** transactional sales data, **performs exploratory data analysis (EDA)**, **engineers business KPIs**, and **exports everything** into a Power BI dashboard for interactive reporting.

| 🎯 Goal | ✅ Result |
|:--------|:---------|
| Clean transactional data | No duplicates, no broken fields |
| Analyze sales performance | Revenue, profit, trends, categories |
| Understand customers | Segmentation by value and behavior |
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

### 🔵 Power BI Dashboard
- 📌 KPI cards (Revenue, Orders, AOV, Profit)
- 📉 Time-series sales trends
- 🏆 Top products & categories
- 🌍 Regional performance breakdown
- 🎛️ Interactive slicers for custom exploration

---

## 🛠️ Installation

### 1. Clone the Repository

git clone https://github.com/Abdalla820/E-Commerce-Sales.git
cd E-Commerce-Sales

2. Install Dependencies
pip install pandas numpy matplotlib seaborn

3. Requirements
Python 3.8+

pandas

numpy

matplotlib

seaborn

Power BI Desktop (free) — to open the .pbix dashboard

🚀 Usage
Basic Run
bash
python E-Commerce.py
Custom File Path
bash
python E-Commerce.py path/to/your/data.csv

Folder Structure

📦 E-Commerce-Sales
 ┣ 📜 E-Commerce.py
 ┣ 📜 E-Commerce.pbix
 ┣ 📜 E-Commerce.pptx 
 ┣ 📂 Dataset
 ┃ ┗ 📄 ecommerce_sales.csv
 ┣ 📂 analysis_output
 ┃ ┣ 📄 cleaned_ecommerce_data.csv   ← generated
 ┃ ┣ 📄 summary_by_product.csv       ← generated
 ┃ ┣ 📄 summary_by_category.csv      ← generated
 ┃ ┣ 📄 summary_monthly.csv          ← generated
 ┃ ┗ 📄 *.png                        ← generated charts
 ┗ 📄 README.md                      ← this file

🧠 How It Works

📥 Step 1 — Load & Inspect
Reads the CSV/Excel file using pandas.

Prints shape, dtypes, missing %, and duplicate count.

Standardizes column names for consistency.

🧹 Step 2 — Clean the Data
Issue	Action
Duplicate rows	Remove
Missing numeric values	Median imputation
Missing categorical values	Fill with "Unknown"
Invalid dates	Coerce to NaT
Non-numeric sales fields	Coerce to NaN → median

⚙️ Step 3 — Feature Engineering
Extracts Year, Month, Quarter, Day of Week

Computes Profit Margin and Unit Price

Creates time-series groupings for trends

📊 Step 4 — KPIs & Statistics
Calculates: Total Revenue, Total Profit, Total Orders, AOV, Avg Margin

Prints plain-English summary of business performance

📈 Step 5 — Visualize
Generates a set of charts saved to analysis_output/:

Monthly sales trend

Top 10 products by revenue

Sales by category

Regional performance

Profit distribution

Correlation heatmap

💾 Step 6 — Output
Saves cleaned_ecommerce_data.csv

Saves summary tables for Power BI

Exports all charts as .png

📊 Sample Output
============================================================
  E-COMMERCE SALES ANALYSIS — START
============================================================
[INFO] Loading dataset: Dataset/ecommerce_sales.csv
[INFO] Loaded shape: (9994, 21)
[INFO] Removed 0 duplicate rows
[INFO] Cleaned shape: (9994, 25)

===== KEY PERFORMANCE INDICATORS =====
         total_revenue: 2,297,200.86
          total_profit: 286,397.02
          total_orders: 5,009
        avg_order_value: 458.61
       avg_profit_margin: 0.12

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
🎯 Key Design Decisions
✅ Principle	💡 Why
Clean before analyzing	Garbage in → garbage out
Preserve real outliers	High-value orders are real, not errors
Median for numerics	Robust to skew in sales data
Separate BI exports	Power BI loads faster with summary tables
Modular functions	Each step is reusable and testable
📁 Output Files
📄 File	📝 Description
cleaned_ecommerce_data.csv	Cleaned dataset ready for analysis
summary_by_product.csv	Revenue per product
summary_by_category.csv	Sales & profit per category
summary_monthly.csv	Monthly sales totals
*.png	All EDA visualizations
E-Commerce.pbix	Interactive Power BI dashboard
🧰 Tech Stack
<div align="center">
https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white
https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white
https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white
https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=python&logoColor=white
https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=python&logoColor=white
https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black

</div>
🤝 Contributing
Contributions are welcome! Feel free to:

🍴 Fork the repository

🌿 Create a feature branch

💾 Commit your changes

🚀 Push and open a Pull Request

📜 License
This project is licensed under the MIT License — free to use and modify.

📧 Contact
<div align="center">
Abdalla Mohamed Farouk

https://img.shields.io/badge/Email-D14836?style=flat-square&logo=gmail&logoColor=white
https://img.shields.io/badge/LinkedIn-0077B5?style=flat-square&logo=linkedin&logoColor=white
https://img.shields.io/badge/GitHub-100000?style=flat-square&logo=github&logoColor=white

</div>
<div align="center">
⭐ If you found this project useful, please give it a star!
Made with ❤️ and Python

</div>
