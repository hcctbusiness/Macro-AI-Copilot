# HCCT Group – Macro-AI-Copilot Tools

End-to-end macro, machine learning, and multi-asset allocation system in Python.

<img width="1200" alt="Macro-AI-Copilot Header" src="https://github.com/user-attachments/assets/6735c336-ee1d-4f4d-b5b5-82bcb29753b1" />

## Overview

Macro-AI-Copilot builds a simple “macro radar” that links global economic data and news sentiment to portfolio allocation decisions.  
It is designed as an educational, research-friendly project for experimenting with macro signals, feature engineering, and ML-driven allocation.

Core ideas:

- Pull macro time series (growth, inflation, financial conditions) from FRED and related sources.
- Combine these with a small, labeled news sentiment sample.
- Learn a compact macro state and map it into expected returns and risk for multiple asset classes.
- Construct and backtest simple allocation rules on top of these macro signals.

## Features

- Data pipeline for downloading and preprocessing macro indicators from FRED.  
- Toy `news_headlines.csv` to demonstrate text-based macro sentiment signals.
- Modular feature engineering and model training in `src/`.  
- Simple multi-asset allocation and backtesting examples.  
- Clear structure so users can plug in their own data, models, and portfolio rules.

## Project structure

- `src/` – core library (data loading, feature engineering, models, allocation, backtests).  
- `data/raw/` – input data such as `news_headlines.csv`.  
- `run_macro_ai_copilot.py` – main script that runs the end-to-end demo pipeline.  
- `requirements.txt` – Python dependencies.  
- `README.md` – project description and usage.

## Installation

```bash
git clone https://github.com/hcctbusiness/Macro-AI-Copilot.git
cd Macro-AI-Copilot
python -m venv .venv
```
## On Windows:
```bash
.venv\Scripts\activate
```

## On macOS/Linux:
```bash
source .venv/bin/activate
```

## Then install dependencies:
```bash
pip install -r requirements.txt
```

## Quick start
```bash
1. Add your FRED API key as an environment variable.

PowerShell:
$env:FRED_API_KEY="YOUR_FRED_API_KEY"

Bash (macOS/Linux):
export FRED_API_KEY="YOUR_FRED_API_KEY"

2. (Optional) Inspect or edit data/raw/news_headlines.csv to adjust the toy news sample.
3. Run the end-to-end demo:
```

```bash
python run_macro_ai_copilot.py
```
This will download macro data, merge it with news sentiment, train simple models, and produce basic allocation and performance outputs.

## Intended use
This repository is for learning, prototyping, and research only, not live trading or investment advice.[web:343]
Use it as a starting point to explore macro data, experiment with ML models, and design your own allocation logic.

## License
Released under the MIT License.

