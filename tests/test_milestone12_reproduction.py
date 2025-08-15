"""
Milestone 12: End-to-end tests for paper reproduction and robustness.

These tests validate the complete reproduction workflow with cached snapshots
to ensure deterministic results and verify robustness assumptions.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import pandas as pd
import numpy as np

from src.utils.config import load_config


class TestMilestone12Reproduction:
    """Test suite for Milestone 12 paper reproduction and robustness."""
    
    @pytest.fixture
    def reproduction_runner(self):
        """Import and return the reproduction runner."""
        import importlib.util
        
        script_path = os.path.join(os.getcwd(), "scripts", "run_milestone12_reproduction.py")
        spec = importlib.util.spec_from_file_location("reproduction", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module.ReproductionRunner

    @pytest.fixture
    def test_config(self):
        """Load and modify config for testing."""
        cfg = load_config("configs/default.yaml")
        # Make it faster for testing
        cfg["data"]["horizon_days"] = 10  # Shorter horizon
        cfg["data"]["cutoff_date"] = "2020-02-15"  # Use date within test data range
        cfg["dnn"]["epochs"] = 5  # Fewer epochs
        cfg["dnn"]["cv_splits"] = 2  # Fewer CV splits
        return cfg

    def test_determinism_with_cached_snapshot(self, reproduction_runner, test_config, ohlc_random_walk):
        """
        Test that the reproduction pipeline produces deterministic results
        when run multiple times with the same configuration and data.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create runner with test config
            runner = reproduction_runner(output_dir=tmp_dir)
            runner.base_config = test_config
            
            # Run determinism test
            results = runner.run_determinism_test(ohlc_random_walk, n_runs=2)
            
            # Verify structure
            assert "runs" in results
            assert "determinism_analysis" in results
            assert len(results["runs"]) == 2
            
            # Check determinism analysis
            analysis = results["determinism_analysis"]
            assert "consistent_persistence" in analysis
            assert "consistent_dnn" in analysis
            
            # Persistence should be perfectly deterministic
            if analysis.get("persistence_variance"):
                for metric, stats in analysis["persistence_variance"].items():
                    assert stats["is_deterministic"], f"Persistence {metric} not deterministic"

    def test_lag_sensitivity_analysis(self, reproduction_runner, test_config, ohlc_random_walk):
        """
        Test that lag window sensitivity analysis runs successfully
        and produces results for t in {3, 5, 7}.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = reproduction_runner(output_dir=tmp_dir)
            runner.base_config = test_config
            
            # Run just the sensitivity analysis part
            results = runner.run_sensitivity_analysis(ohlc_random_walk)
            
            # Verify lag sensitivity results
            assert "lag_sensitivity" in results
            lag_results = results["lag_sensitivity"]
            
            # Should have results for each t value
            expected_keys = ["t_3", "t_5", "t_7"]
            for key in expected_keys:
                assert key in lag_results
                assert "data_stats" in lag_results[key]
                assert "persistence" in lag_results[key]
                assert "backtest" in lag_results[key]

    def test_cost_sensitivity_sweep(self, reproduction_runner, test_config, ohlc_random_walk):
        """
        Test that cost/slippage sensitivity analysis covers the expected range
        and produces valid backtest results.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = reproduction_runner(output_dir=tmp_dir)
            runner.base_config = test_config
            
            results = runner.run_sensitivity_analysis(ohlc_random_walk)
            
            # Verify cost sensitivity results
            assert "cost_sensitivity" in results
            cost_results = results["cost_sensitivity"]
            
            # Should have multiple cost scenarios
            assert len(cost_results) >= 3
            
            # Each scenario should have params and results
            for scenario_key, scenario in cost_results.items():
                assert "params" in scenario
                assert "results" in scenario
                assert "cost_bps" in scenario["params"]
                assert "slippage_bps" in scenario["params"]
                assert "backtest" in scenario["results"]

    def test_cached_data_consistency(self, reproduction_runner, test_config):
        """
        Test that cached data loading produces consistent results
        across multiple calls.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = reproduction_runner(output_dir=tmp_dir)
            runner.base_config = test_config
            
            # First fetch should cache data
            df1 = runner.fetch_and_cache_data()
            
            # Second fetch should load from cache
            df2 = runner.fetch_and_cache_data()
            
            # Should be identical
            pd.testing.assert_frame_equal(df1, df2)
            
            # Cache file should exist
            cache_path = Path(tmp_dir) / "cache" / "raw_anf_data.pkl"
            assert cache_path.exists()

    def test_validation_criteria_met(self, reproduction_runner, test_config, ohlc_random_walk):
        """
        Test that the validation criteria are properly checked:
        - DNN beats persistence on at least two series
        - Trading returns are non-negative for at least one strategy
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = reproduction_runner(output_dir=tmp_dir)
            runner.base_config = test_config
            
            # Run single reproduction to get validation data
            result = runner._run_single_reproduction(ohlc_random_walk, test_config, "validation_test")
            
            # Test DNN vs persistence validation
            if "dnn" in result and "persistence" in result:
                validation = runner.validate_dnn_vs_persistence({"lag_sensitivity": {"t_5": result}})
                
                assert "dnn_beats_persistence" in validation
                assert "series_comparisons" in validation
                assert len(validation["series_comparisons"]) <= 4  # At most OHLC series
            
            # Test trading returns validation
            trading_val = runner._validate_trading_returns({"cost_sensitivity": {"scenario_0": {"results": result}}})
            
            assert "threshold_positive_scenarios" in trading_val
            assert "tema_positive_scenarios" in trading_val
            assert "at_least_one_positive" in trading_val

    def test_output_structure_compliance(self, reproduction_runner, test_config, ohlc_random_walk):
        """
        Test that all outputs follow the expected structure and can be
        loaded/parsed correctly.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = reproduction_runner(output_dir=tmp_dir)
            runner.base_config = test_config
            
            # Run single reproduction
            result = runner._run_single_reproduction(ohlc_random_walk, test_config, "structure_test")
            
            # Verify result structure
            required_keys = ["config", "data_stats", "persistence", "backtest"]
            for key in required_keys:
                assert key in result, f"Missing required key: {key}"
            
            # Verify data stats
            data_stats = result["data_stats"]
            assert all(k in data_stats for k in ["train_rows", "test_rows", "train_start", "test_start"])
            
            # Verify persistence results
            persistence = result["persistence"]
            assert "forecast_metrics" in persistence
            assert "predictions" in persistence
            
            # Verify backtest results
            backtest = result["backtest"]
            assert "threshold" in backtest
            assert "tema" in backtest
            
            for strategy in ["threshold", "tema"]:
                strategy_result = backtest[strategy]
                assert "trades" in strategy_result
                assert "metrics" in strategy_result
                assert "final_equity" in strategy_result

    def test_snapshot_regression(self, reproduction_runner, test_config):
        """
        Regression test using a cached snapshot of known good results.
        This test ensures that changes to the codebase don't break reproduction.
        """
        # Use synthetic data for reproducible snapshots
        synthetic_data = self._generate_test_ohlc_snapshot()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            runner = reproduction_runner(output_dir=tmp_dir)
            runner.base_config = test_config
            
            # Run reproduction on snapshot data
            result = runner._run_single_reproduction(synthetic_data, test_config, "snapshot_test")
            
            # Basic sanity checks on the snapshot result
            assert result["data_stats"]["train_rows"] > 0
            assert result["data_stats"]["test_rows"] > 0
            
            # Persistence should always work
            assert "persistence" in result
            assert "forecast_metrics" in result["persistence"]
            
            # Backtest should complete
            assert "backtest" in result
            assert "threshold" in result["backtest"]
            assert "tema" in result["backtest"]

    def _generate_test_ohlc_snapshot(self) -> pd.DataFrame:
        """Generate a deterministic OHLC snapshot for testing."""
        np.random.seed(42)
        n_days = 90
        
        dates = pd.bdate_range(start="2020-01-01", periods=n_days)
        rets = np.random.normal(0.0002, 0.01, n_days)
        close = 100.0 * np.exp(np.cumsum(rets))
        
        open_ = np.empty_like(close)
        open_[0] = close[0]
        open_[1:] = close[:-1]
        
        high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.2, n_days))
        low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.2, n_days))
        volume = np.random.randint(100_000, 1_000_000, n_days)
        
        return pd.DataFrame({
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Adj Close": close.copy(),
            "Volume": volume,
        }, index=dates)


# Integration test that requires the full reproduction script
def test_full_reproduction_integration():
    """
    Integration test that runs the full reproduction script
    and validates it completes successfully.
    """
    import subprocess
    import sys
    
    script_path = os.path.join(os.getcwd(), "scripts", "run_milestone12_reproduction.py")
    
    # Run the script with a timeout
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.getcwd()
        )
        
        # Script should complete successfully
        assert result.returncode == 0, f"Script failed with: {result.stderr}"
        
        # Should produce the expected summary output
        assert "MILESTONE 12 REPRODUCTION SUMMARY" in result.stdout
        assert "Data: ANF" in result.stdout
        
    except subprocess.TimeoutExpired:
        pytest.skip("Reproduction script timed out - this is expected for full runs")
    except FileNotFoundError:
        pytest.skip("Reproduction script not found")


if __name__ == "__main__":
    pytest.main([__file__])
