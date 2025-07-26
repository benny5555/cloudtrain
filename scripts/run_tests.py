#!/usr/bin/env python3
"""Test runner script for CloudTrain.

This script provides a convenient way to run different types of tests
with appropriate configurations and reporting.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} completed successfully")
        return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="CloudTrain Test Runner")
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all"],
        default="unit",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Run tests in verbose mode"
    )
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument(
        "--provider",
        choices=["aws", "azure", "gcp", "mock"],
        help="Run tests for specific provider only",
    )
    parser.add_argument(
        "--parallel", "-n", type=int, help="Number of parallel test processes"
    )

    args = parser.parse_args()

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add coverage if requested
    if args.coverage:
        cmd.extend(
            [
                "--cov=src/cloudtrain",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
            ]
        )

    # Add parallel execution if requested
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    # Configure test selection based on type
    if args.type == "unit":
        cmd.extend(["-m", "unit"])
        if args.fast:
            cmd.extend(["-m", "unit and not slow"])
    elif args.type == "integration":
        cmd.extend(["-m", "integration"])
        if not args.fast:
            cmd.extend(["-m", "integration"])
        else:
            cmd.extend(["-m", "integration and not slow"])
    elif args.type == "all":
        if args.fast:
            cmd.extend(["-m", "not slow"])

    # Add provider-specific filtering
    if args.provider:
        if args.type != "all":
            cmd[-1] += f" and {args.provider}"
        else:
            cmd.extend(["-m", args.provider])

    # Add test discovery paths
    cmd.extend(["src/", "tests/", "--tb=short"])

    # Run the tests
    success = run_command(cmd, f"CloudTrain {args.type} tests")

    if not success:
        sys.exit(1)

    print(f"\n🎉 All {args.type} tests passed!")

    if args.coverage:
        print("\n📊 Coverage report generated in htmlcov/index.html")


if __name__ == "__main__":
    main()
