# Production & Inventory Optimization Agent 

## Project Overview
This project implements an AI-assisted agent for monitoring and optimizing production and inventory in a supply chain context.  
The agent analyzes inventory data, detects anomalies, and supports decision-making for restocking and production planning.

This project is developed as part of the course **Advanced Programming and Data Visualization**.

---

## Project Goals
- Monitor inventory and sales data
- Detect stockout and overstock situations
- Support restocking decisions
- Generate basic reports and visualizations

---

## Project Structure
```
production_inventory_agent/
├── config/
├── database_creation/
├── scripts/
│   ├── LLM.py
│   └── setup_database.py
├── src/
│   └── agent_tools/
│       ├── __init__.py
│       ├── database.py
│       ├── analysis.py
│       ├── visualization.py
│       └── reports.py
└── README.md
```

---

## Database
The project uses an inventory database containing:
- product
- date
- stock_level
- sold_units

The database can be initialized using:
```bash
python scripts/setup_database.py
```
---

## More details:

### Natural Language Interface
- Chat with the agent
- Intelligent query understanding and intent detection

### Analytics & Insights
- Stock Analysis: Real-time inventory levels across all SKUs
- Sales Metrics: Category-level and SKU-level performance tracking
- Demand Classification: Automatic categorization (HIGH/MODERATE/LOW/VERY LOW)
- Revenue Tracking: Accurate revenue calculations without duplication

### Demand Forecasting
- Multiple forecasting methods (ARIMA, Prophet)
- Configurable forecast horizons (7, 14, 30+ days)
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
- "How much did cosmetics sell this month?"
- "Sales performance for skincare last week"

### Forecasting
- "Forecast demand for skincare"
- "What will sales be in 2 weeks for haircare?"
- "Predict next month's demand for cosmetics"

### Anomaly Detection
- "Are there any problems with haircare?"
- "Detect anomalies in skincare"
- "Show me stock issues"

### Restocking
- "Should I order skincare?"
- "Which products need restocking?"
- "Restock recommendations"

### Visualizations
- "Show me the haircare chart"
- "Display stock evolution for cosmetics"
- "Graph sales trends"

### Global Status
- "What's the status?"
- "Give me a summary"
- "Overall supply chain report"



