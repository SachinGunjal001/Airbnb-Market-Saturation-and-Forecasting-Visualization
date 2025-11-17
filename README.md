# Airbnb-Driven-Market-Saturation-and-Forecasting-Visualization-Dashboard - NYC (Full Project Walkthrough)

## 📊 Objective

The goal of this project is to analyze Airbnb listing data in New York City to:

* Identify pricing patterns by region, neighborhood, and room type
* Understand how Airbnb presence affects long-term rental prices
* Forecast future price trends
* Generate insights for city planners, investors, and policy makers

---

## 🎯 Key Contributions

* Built a full ETL pipeline with Python and PostgreSQL
* Cleaned and enriched raw Airbnb listing data
* Performed EDA and statistical modeling to extract insights
* Visualized key KPIs using Tableau dashboards
* Forecasted monthly average price using regression models
* Wrote complex SQL queries with spatial joins to enrich data

---

## 🧱 Repository Structure

```bash
Airbnb-NYC-Analysis/
├── data/
│   ├── AB_NYC_2019.csv                       # Raw dataset
│   ├── AB_NYC_2019_cleaned.csv               # Cleaned output from ETL
│   ├── price_distribution.png                # Output plot from EDA
├── scripts/
│   ├── etl_pipeline.py                       # Data cleaning and feature generation
│   ├── eda_modeling.py                       # EDA, regression, and forecasting
├── sql/
│   └── spatial_queries.sql                   # SQL joins and aggregation
├── README.md                                 # Project documentation
└── requirements.txt                          # Python dependencies
└── Final Tableau Dashboard Project link      # Data Visualization - Tableau Dashboard
```

---

## 📦 Requirements

Install all dependencies via pip:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
pandas==1.5.3
numpy==1.21.6
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.3.0
statsmodels==0.13.5
```

---

## 🚀 Installation & Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Airbnb-NYC-Analysis.git
cd Airbnb-NYC-Analysis
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run ETL pipeline to clean and preprocess data:

```bash
python scripts/etl_pipeline.py
```

4. Perform EDA and generate plots:

```bash
python scripts/eda_modeling.py
```

5. Use Tableau to load `AB_NYC_2019_cleaned.csv` and replicate dashboards

---

## 🔍 Modules Breakdown

### `etl_pipeline.py`

* Removes nulls
* Filters invalid price listings
* Creates new features (e.g. month, price/min night)

### `eda_modeling.py`

* Generates boxplots and distribution plots
* Fits a linear regression model
* Runs time-series forecasting

### `spatial_queries.sql`

* Performs joins with zoning and census data
* Aggregates metrics for neighborhoods and boroughs

---

## 📈 KPIs Tracked

* Average Monthly Price
* Listings per Borough
* Top Room Types
* Availability 365
* Repeat Reviewers
* Regression Coefficients

---

## 📊 Dashboard Highlights (Tableau)

* Map of listings by neighborhood
* Filters by price, availability, room type
* KPIs and bar charts by borough and zip code

---

## ❓ Business Questions Answered

* Which boroughs have the highest and lowest prices?
* What listing types generate the most income?
* Are there seasonal price variations by month?
* How does availability vary by region?
* Which areas might be over/under saturated?

---

## 🔍 Example Output

* Boxplot of prices by borough (removing \$500+ outliers)
* Linear regression summary with R-squared and coefficients
* Predicted trendline of price vs. month
* Aggregated SQL summary by region

---

## 👩‍💻 Author

**Anushka Patil**
Data Analyst | Python | SQL | PostGreSQL | Tableau | EDA Data Modeling | ETL Pipeline
[LinkedIn](https://www.linkedin.com/in/anushkapatil272000/)
[Tableau Dashboard Project link](https://public.tableau.com/views/AirbnbMarketAnalysisforNYC/Dashboard1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## 💬 Contributions Welcome

Fork the repo, create issues, and submit PRs! This project is open for enhancement and extension with external datasets (e.g. rent control, census, and API integrations).
