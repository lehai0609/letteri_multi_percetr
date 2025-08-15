Milestone 5 — Walking skeleton (persistence forecaster)
======================================================

Overview
--------
This milestone implements a minimal end-to-end pipeline (walking skeleton) that produces 30-day OHLC forecasts using a persistence forecaster, generates strategy signals, backtests them on the actual test window, and computes forecast + backtest metrics.

What I added
- src/forecasting/persistence.py
  - persistence_forecast(train_df, test_index, method="last"|"drift")
  - forecast_from_split(train_df, test_df)
- src/metrics/forecast.py
  - mae, rmse, mape, explained_variance_score
  - compute_forecast_metrics(actual_df, pred_df) -> per-series metrics + summary
- scripts/run_milestone5.py
  - End-to-end runner (synthetic fixture -> split -> persistence forecast -> signals -> backtest -> metrics)
  - Inserts repo root onto sys.path so it can be executed directly from the project root

How to run (from repo root)
1. (Optional) Create and activate a virtualenv:
   - Windows (cmd): python -m venv .venv && .venv\Scripts\activate
2. Install project deps if needed:
   - python -m pip install -r requirements.txt
   - or install pytest only: python -m pip install pytest
3. Run tests:
   - python -m pytest -q
4. Run the Milestone 5 runner:
   - python scripts/run_milestone5.py

Runner outputs (saved to outputs/milestone5/)
- predicted_ohlc.csv
- forecast_metrics.json
- threshold_backtest_metrics.json
- tema_backtest_metrics.json
- threshold_trades.json
- tema_trades.json
- equity_threshold.csv
- equity_tema.csv

Notes / Caveats
- split_by_cutoff uses trading-calendar utils and may raise "missing trading day" for synthetic fixtures; the runner falls back to a positional split in that case so the walking skeleton runs offline without external calendars.
- The persistence forecaster uses only historical train_df values (no data leakage).
- Triple-TEMA can produce zero trades on some synthetic seeds — this is expected for low-signal synthetic data.
- When running the script directly, run from the repository root to avoid import issues. The runner inserts the repo root to sys.path to mitigate this.

Suggested commit message
- Title: feat(milestone5): persistence forecaster, forecast metrics, and end-to-end runner
- Body:
  - Added persistence forecaster and forecast metrics utilities.
  - Added scripts/run_milestone5.py to produce persistence forecasts and run threshold/TEMA backtests on synthetic data.
  - Runner is runnable directly from the repo root.

Suggested README snippet (copy into README.md)
- Milestone 5: Walking skeleton
  - Run tests: `python -m pytest -q`
  - Run persistence-runner: `python scripts/run_milestone5.py`
  - Outputs are saved to `outputs/milestone5/`. Run from project root.

Next recommended step
- Milestone 6: Implement feature windowing:
  - Add `src/features/window.py` with:
    - build_xy(series_scaled, t) -> (X, y)
    - last_window(series_scaled, t) -> np.ndarray of shape (t,)
  - Unit tests: on series [1..10], t=5 => X shape (4,5), y shape (4,) and alignment check

Contact
-------
If you want, I can:
- Add the README snippet directly to README.md
- Add unit tests for the persistence forecaster and forecast metrics
- Proceed to Milestone 6 and implement `src/features/window.py`
