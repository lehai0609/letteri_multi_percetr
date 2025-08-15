#!/usr/bin/env python3
"""
Convenient CLI wrapper for paper reproduction (Milestone 12).

Usage:
    python scripts/reproduce_paper.py                    # Full reproduction
    python scripts/reproduce_paper.py --quick            # Quick test with synthetic data
    python scripts/reproduce_paper.py --sensitivity-only # Only sensitivity analysis
    python scripts/reproduce_paper.py --determinism-only # Only determinism check
"""
from __future__ import annotations

import argparse
import sys
import os
import logging
from pathlib import Path

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import the reproduction runner
from scripts.run_milestone12_reproduction import ReproductionRunner

# Import synthetic data generator for quick tests
from scripts.run_milestone5 import generate_synthetic_ohlc


def setup_logging(log_level: str = "INFO"):
    """Setup logging for the CLI."""
    numeric_level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def run_quick_test(output_dir: str = "outputs/milestone12_quick"):
    """Run a quick reproduction test using synthetic data."""
    print("Running quick reproduction test with synthetic data...")
    
    runner = ReproductionRunner(output_dir=output_dir)
    
    # Generate synthetic data
    synthetic_df = generate_synthetic_ohlc(n_days=90, start="2020-01-01", seed=1234)
    
    # Run a single reproduction with adjusted dates for synthetic data
    config = runner.base_config.copy()
    config["data"]["horizon_days"] = 15  # Shorter for quick test
    config["data"]["cutoff_date"] = "2020-02-15"  # Use date within synthetic range
    
    result = runner._run_single_reproduction(synthetic_df, config, "quick_test")
    
    print("\nQuick test results:")
    print(f"- Train rows: {result['data_stats']['train_rows']}")
    print(f"- Test rows: {result['data_stats']['test_rows']}")
    print(f"- Persistence MAE: {result['persistence']['forecast_metrics'].get('summary', {}).get('mae', 'N/A')}")
    print(f"- Threshold final equity: ${result['backtest']['threshold']['final_equity']:.2f}")
    print(f"- TEMA final equity: ${result['backtest']['tema']['final_equity']:.2f}")
    
    return result


def run_sensitivity_only(output_dir: str = "outputs/milestone12_sensitivity"):
    """Run only the sensitivity analysis."""
    print("Running sensitivity analysis only...")
    
    runner = ReproductionRunner(output_dir=output_dir)
    df = runner.fetch_and_cache_data()
    
    results = runner.run_sensitivity_analysis(df)
    
    print("\nSensitivity analysis completed:")
    print(f"- Lag windows tested: {list(results['lag_sensitivity'].keys())}")
    print(f"- Cost scenarios tested: {len(results['cost_sensitivity'])}")
    
    return results


def run_determinism_only(output_dir: str = "outputs/milestone12_determinism"):
    """Run only the determinism verification."""
    print("Running determinism verification only...")
    
    runner = ReproductionRunner(output_dir=output_dir)
    df = runner.fetch_and_cache_data()
    
    results = runner.run_determinism_test(df, n_runs=3)
    
    print("\nDeterminism test completed:")
    analysis = results["determinism_analysis"]
    print(f"- Persistence consistent: {analysis.get('consistent_persistence', 'N/A')}")
    print(f"- DNN consistent: {analysis.get('consistent_dnn', 'N/A')}")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Paper reproduction CLI for Milestone 12")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick test with synthetic data")
    parser.add_argument("--sensitivity-only", action="store_true",
                       help="Run only sensitivity analysis")
    parser.add_argument("--determinism-only", action="store_true",
                       help="Run only determinism verification")
    parser.add_argument("--output-dir", default="outputs/milestone12",
                       help="Output directory for results")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    try:
        if args.quick:
            run_quick_test(args.output_dir + "_quick")
        elif args.sensitivity_only:
            run_sensitivity_only(args.output_dir + "_sensitivity")
        elif args.determinism_only:
            run_determinism_only(args.output_dir + "_determinism")
        else:
            # Full reproduction
            print("Starting full paper reproduction...")
            runner = ReproductionRunner(output_dir=args.output_dir)
            report = runner.run_complete_reproduction()
            
            print(f"\nFull reproduction completed successfully!")
            print(f"Results saved to: {args.output_dir}")
            
            # Print key findings
            validation = report["validation"]
            print(f"\nKey findings:")
            print(f"- DNN beats persistence: {validation['dnn_vs_persistence'].get('dnn_beats_persistence', 'N/A')}")
            print(f"- Positive trading returns: {validation['trading_returns'].get('at_least_one_positive', 'N/A')}")
    
    except Exception as exc:
        logging.error(f"Reproduction failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
