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
│   ├── server.py
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
