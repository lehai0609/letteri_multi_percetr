# Milestone 12: Paper Reproduction and Robustness Guide

This guide walks you through implementing and running the complete paper reproduction with robustness validation.

## Overview

Milestone 12 validates the entire system by:
1. **Running on real ANF data** with the exact date ranges from your config
2. **Sensitivity analysis** on key parameters (lag window t, transaction costs)
3. **Determinism verification** to ensure reproducible results
4. **Performance validation** ensuring DNN beats persistence and trading returns are viable

## Quick Start

### 1. Run Full Reproduction
```bash
python scripts/reproduce_paper.py
```

### 2. Quick Test (Synthetic Data)
```bash
python scripts/reproduce_paper.py --quick
```

### 3. Run Only Sensitivity Analysis
```bash
python scripts/reproduce_paper.py --sensitivity-only
```

### 4. Run Only Determinism Check
```bash
python scripts/reproduce_paper.py --determinism-only
```

## Implementation Architecture

### Main Components

1. **`scripts/run_milestone12_reproduction.py`** - Core reproduction engine
2. **`scripts/reproduce_paper.py`** - CLI wrapper for convenience
3. **`tests/test_milestone12_reproduction.py`** - Test suite with cached snapshots

### Workflow Structure

```
ReproductionRunner
├── fetch_and_cache_data()          # ANF data with disk caching
├── run_sensitivity_analysis()      # t={3,5,7} + cost sweeps  
├── run_determinism_test()          # Multiple runs with same seeds
├── validate_dnn_vs_persistence()   # Performance comparison
└── run_complete_reproduction()     # Orchestrates everything
```

## Sensitivity Analysis Details

### 1. Lag Window Testing (t = 3, 5, 7)
- Tests impact of different lookback windows on forecast accuracy
- Validates that t=5 (paper default) is reasonable
- Each t value gets complete pipeline: train → forecast → backtest

### 2. Transaction Cost Sweep
Tests 5 cost scenarios:
- **No cost**: 0.0 bps commission, 0.0 bps slippage
- **Low cost**: 5.0 bps commission, 2.5 bps slippage  
- **Default**: 10.0 bps commission, 5.0 bps slippage
- **High cost**: 25.0 bps commission, 12.5 bps slippage
- **Very high**: 50.0 bps commission, 25.0 bps slippage

## Validation Criteria

The reproduction **passes** if:

1. **Determinism**: Same config + same seed → identical results
2. **DNN Performance**: DNN beats persistence on ≥2 OHLC series (MAE comparison)
3. **Trading Viability**: ≥1 strategy shows non-negative returns in ≥1 cost scenario
4. **Calendar Alignment**: No weekend/holiday data leakage
5. **Data Quality**: Complete OHLC bars, proper train/test split

## Output Structure

```
outputs/milestone12/
├── reproduction_report.json        # Main summary report
├── reproduction.log                # Detailed execution log
├── cache/
│   └── raw_anf_data.pkl            # Cached ANF data
└── runs/
    ├── determinism_run_0/          # First determinism run
    ├── determinism_run_1/          # Second determinism run  
    ├── lag_t_3/                    # Lag window t=3 results
    ├── lag_t_5/                    # Lag window t=5 results
    ├── lag_t_7/                    # Lag window t=7 results
    ├── costs_0/                    # No-cost scenario
    ├── costs_1/                    # Low-cost scenario
    ├── costs_2/                    # Default-cost scenario
    ├── costs_3/                    # High-cost scenario
    └── costs_4/                    # Very-high-cost scenario
```

Each run directory contains:
- `results.json` - Complete results for that configuration
- `train_data.pkl` / `test_data.pkl` - Cached split data
- `models/` - Trained model artifacts (if DNN training succeeded)
- `grid_search.json` - Hyperparameter search results

## Running Tests

### Unit Tests
```bash
python -m pytest tests/test_milestone12_reproduction.py -v
```

### Quick Integration Test
```bash
python -m pytest tests/test_milestone12_reproduction.py::test_full_reproduction_integration -v
```

### All Milestone 12 Tests
```bash
python -m pytest tests/test_milestone12_reproduction.py
```

## Troubleshooting

### Common Issues

1. **TensorFlow not available**: DNN training will be skipped, persistence-only mode
2. **Insufficient data**: Check date ranges in `configs/default.yaml`
3. **Memory issues**: Reduce `horizon_days` or `n_grid` for testing
4. **Network timeouts**: Cached data will be reused on subsequent runs

### Debug Mode
```bash
python scripts/reproduce_paper.py --log-level DEBUG
```

### Manual Investigation
```python
from scripts.run_milestone12_reproduction import ReproductionRunner

runner = ReproductionRunner()
df = runner.fetch_and_cache_data()
print(f"Data shape: {df.shape}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
```

## Expected Runtime

- **Quick test**: ~30 seconds (synthetic data)
- **Sensitivity only**: ~5-10 minutes (real data, no DNN)
- **Full reproduction**: ~30-60 minutes (depends on DNN grid search)
- **Tests**: ~2-5 minutes

## Success Criteria Validation

The script automatically validates:

### ✅ Data Quality
- ANF data downloaded and cached successfully
- Proper business day calendar (no weekends/holidays)
- Clean OHLC bars with no missing values
- Correct train/test split at cutoff date

### ✅ Determinism  
- Multiple runs with same seeds produce identical results
- Persistence forecaster is perfectly deterministic
- DNN results consistent within numerical precision

### ✅ Performance Benchmarks
- DNN achieves lower MAE than persistence on ≥2 series
- At least one trading strategy shows positive returns
- Strategies generate reasonable trade counts

### ✅ Robustness
- Results stable across lag window variations
- Performance degrades gracefully with higher transaction costs
- System handles edge cases (market gaps, extreme volatility)

## Next Steps After Milestone 12

If reproduction succeeds:
1. **Milestone 13**: Real-time deployment preparation
2. **Milestone 14**: Model monitoring and drift detection  
3. **Extensions**: Multi-asset portfolios, alternative strategies

If reproduction fails:
1. Check validation criteria in the final report
2. Review logs for specific failure points
3. Consider parameter tuning or data quality improvements
