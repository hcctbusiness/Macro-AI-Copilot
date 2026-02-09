# Macro-AI-Copilot

End-to-end macro, machine learning, and multi-asset allocation system in Python.

<img width="2752" height="1536" alt="Post 16 - Photo1" src="https://github.com/user-attachments/assets/6735c336-ee1d-4f4d-b5b5-82bcb29753b1" />

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
- Toy news-headline sentiment CSV to demonstrate text-based macro signals.  
- Modular feature engineering and model training in `src/`.  
- Simple multi-asset allocation and backtesting examples.  
- Clear structure so users can plug in their own data, models, and portfolio rules.

## Project structure

- `src/` – core library (data, features, models, allocation, backtests).  
- `data/raw/` – input data such as `news_headlines.csv`.  
- `run_macro_ai_copilot.py` – main entry point that runs the end-to-end demo pipeline.  
- `requirements.txt` – Python dependencies.  
- `README.md` – project description and usage.

## Installation

```bash
git clone https://github.com/hcctbusiness/Macro-AI-Copilot.git
cd Macro-AI-Copilot
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

