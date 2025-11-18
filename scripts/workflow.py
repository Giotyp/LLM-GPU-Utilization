#!/usr/bin/env python3
"""
Workflow orchestrator for running multiple GPU profiling tests.
Automatically runs tests for multiple model-workload combinations.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import List, Dict
import time


class ProfileWorkflow:
    def __init__(self, config_dir: str, results_dir: str = ".results"):
        self.config_dir = Path(config_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.script_dir = Path(__file__).parent

    def list_configs(self) -> List[Path]:
        """Find all workload-model configs."""
        configs = list(self.config_dir.glob("**/*.json"))
        return sorted(configs)

    def run_profile(self, config_path: Path) -> bool:
        """Run profiling for a single config."""
        print(f"\n{'='*70}")
        print(f"Running profile: {config_path.name}")
        print(f"{'='*70}")

        cmd = [
            "bash",
            str(self.script_dir / "run_profile.sh"),
            str(config_path),
            str(self.results_dir),
        ]

        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Profile failed with return code {e.returncode}")
            return False
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    def analyze_results(self, results_dirs: List[Path]) -> bool:
        """Analyze results from completed runs."""
        print(f"\n{'='*70}")
        print("Running comparative analysis...")
        print(f"{'='*70}\n")

        cmd = [
            "python3",
            str(self.script_dir / "analyze_results.py"),
            *[str(d) for d in results_dirs],
        ]

        try:
            result = subprocess.run(cmd, check=False, capture_output=False, text=True)
            return True
        except Exception as e:
            print(f"ERROR: Analysis failed - {e}")
            return False

    def run_all(self, config_names: List[str] = None):
        """Run profiling for all (or specified) configs."""
        configs = self.list_configs()

        # Filter configs if specified
        if config_names:
            configs = [
                c for c in configs if any(name in c.name for name in config_names)
            ]

        if not configs:
            print("No configs found!")
            return

        print(f"Found {len(configs)} config(s):")
        for i, cfg in enumerate(configs, 1):
            print(f"  {i}. {cfg.name}")

        print("\n" + "=" * 70)
        print("STARTING PROFILING WORKFLOW")
        print("=" * 70)

        completed = []
        failed = []

        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Processing {config.name}...")

            if self.run_profile(config):
                completed.append(config)
            else:
                failed.append(config)

            # Small delay between runs
            if i < len(configs):
                time.sleep(5)

        # Print summary
        print(f"\n\n{'='*70}")
        print("PROFILING SUMMARY")
        print(f"{'='*70}")
        print(f"Completed: {len(completed)}/{len(configs)}")
        for cfg in completed:
            print(f"  ✓ {cfg.name}")

        if failed:
            print(f"\nFailed: {len(failed)}")
            for cfg in failed:
                print(f"  ✗ {cfg.name}")

        # Run comparative analysis
        if completed:
            result_dirs = list(self.results_dir.glob("*/"))
            if result_dirs:
                self.analyze_results(sorted(result_dirs))

        print(f"\nResults saved to: {self.results_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="GPU Profiling Workflow Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow.py config/workload-model/
      Run all configs in the directory
  
  python workflow.py config/workload-model/ --results benchmarks/
      Save results to custom directory
  
  python workflow.py config/workload-model/ --filter 1p 100p
      Only run configs matching '1p' or '100p'
""",
    )

    parser.add_argument(
        "config_dir", help="Directory containing workload-model configs"
    )
    parser.add_argument(
        "--results",
        "-r",
        default=".results",
        help="Results directory (default: .results)",
    )
    parser.add_argument(
        "--filter", "-f", nargs="+", help="Filter configs by name patterns"
    )

    args = parser.parse_args()

    workflow = ProfileWorkflow(args.config_dir, args.results)
    workflow.run_all(args.filter)


if __name__ == "__main__":
    main()
