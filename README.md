# Production & Inventory Optimization Agent 

## Project Overview
This project implements an AI-assisted agent for monitoring and optimizing production and inventory in a supply chain context.  
The agent analyzes inventory data, detects anomalies, forecasts demand, and supports decision-making for restocking and production planning.

This project is developed as part of the course **Advanced Programming and Data Visualization**.

---

## Project Goals
- Monitor inventory and sales data
- Detect stockout and overstock situations
- Support restocking decisions
- Forecast demand
- Generate basic reports and visualizations
- Provide a natural-language interface (LLM-style CLI)

---

## Project Structure
```text
production_inventory_agent/
├── README.md
├── config/
│   └── config.yaml
├── data/
│   ├── data.csv
│   ├── processed/
│   └── raw/
├── database_creation/
│   ├── enriched_supply_chain_data.csv
│   ├── explanation/
│   └── improvmentbd.py
├── scripts/
│   ├── LLM.py
│   └── setup_database.py
└── src/
    └── agent_tools/
        ├── __init__.py
        ├── analysis.py
        ├── database.py
        ├── reports.py
        └── visualization.py
```

---

## Database
The agent loads supply-chain data from a CSV file containing:
- product type
- date
- stock_level
- sold_units

---

## CSV location
By default, the project will look for a CSV file in one of the following locations:

- `data/data.csv`
- `database_creation/enriched_supply_chain_data.csv`
- Project root (e.g. `./enriched_supply_chain_data.csv`)


---

## Quickstart (Run Locally)
1) Create and activate a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```
2) Install dependencies
```bash
python -m pip install -U pip
python -m pip install pandas numpy matplotlib seaborn transformers torch
```
3) Ensure the CSV is in the expected location
Example
```bash
cp database_creation/enriched_supply_chain_data.csv data/data.csv
```
4) Run the agent
```bash
python scripts/LLM.py
```

---

## Features

### Natural Language Interface
- Chat with the agent in a terminal
- Intent detection (stock, sales, forecast, anomalies, restock, status, help)


### Analytics and Insight
- Stock Analysis: Real-time inventory levels across all SKUs
- Sales Metrics: Category-level and SKU-level performance tracking
- Demand Classification: Automatic categorization (HIGH/MODERATE/LOW/VERY LOW)
- Revenue Tracking: Accurate revenue calculations without duplication

#### Demand Forecasting
- Multiple forecasting methods (ARIMA, Prophet)
- Configurable forecast horizon (7 / 14 / 30+ days)
- Confidence intervals and trend detection


### Anomaly Detection
- Automated stockout detection
- Overstock identification
- Critical stock level alerts
- Z-score based statistical analysis
- HF-powered anomaly classification

### Visualizations
- Stock evolution charts
- Weekly demand aggregation
- Forecast vs. actual comparisons
- Anomaly highlighting
- Product comparisons

### Hugging Face
- Sentiment analysis for market trends
- Zero-shot classification for anomalies
- AI-powered text generation for recommendations
- Enhanced forecasting with sentiment adjustments

---

## Usage Examples

### Stock Queries
- "What is the stock for skincare?"
- "How many units of cosmetics are left?"
- "Show me haircare inventory"

### Sales Analysis
- "What are the sales for haircare?"
- "Sales performance for skincare last week"

### Forecasting
- "Forecast demand for skincare"
- "Predict next month's demand for cosmetics"

### Anomaly Detection
- "Are there any problems with haircare?"
- "Detect anomalies in skincare"
- "Show me stock issues"

### Restocking
- "Should I order skincare?"
- "Predict next month's demand for cosmetics"

### Visualizations
- "Show me the haircare chart"

### Global Status
- "What's the status?"
- "Give me a summary"
- "Overall supply chain report"

