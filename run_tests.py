#!/usr/bin/env python3
"""
Test runner for Porter.AI with TDD support.
Run different test suites and generate coverage reports.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd: list, description: str) -> int:
    """Run a command and return exit code"""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Porter.AI Test Runner")
    parser.add_argument("--suite", choices=["all", "unit", "integration", "benchmark", "vlm", "capture"],
                       default="all", help="Test suite to run")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--failfast", "-x", action="store_true", help="Stop on first failure")
    parser.add_argument("--markers", "-m", help="Run tests with specific markers")
    parser.add_argument("--watch", action="store_true", help="Watch for changes and re-run tests")
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    # Add fail fast
    if args.failfast:
        cmd.append("-x")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=app", "--cov-report=term-missing", "--cov-report=html"])
    
    # Select test suite
    if args.suite == "all":
        cmd.append("tests/")
        description = "Running all tests"
    elif args.suite == "unit":
        cmd.extend(["-m", "unit", "tests/"])
        description = "Running unit tests"
    elif args.suite == "integration":
        cmd.extend(["-m", "integration", "tests/integration/"])
        description = "Running integration tests"
    elif args.suite == "benchmark":
        cmd.extend(["-m", "benchmark", "tests/"])
        description = "Running performance benchmarks"
    elif args.suite == "vlm":
        cmd.append("tests/vlm_processors/")
        description = "Running VLM processor tests"
    elif args.suite == "capture":
        cmd.append("tests/capture/")
        description = "Running capture tests"
    
    # Add custom markers
    if args.markers:
        cmd.extend(["-m", args.markers])
        description += f" with markers: {args.markers}"
    
    # Watch mode
    if args.watch:
        print("📺 Watch mode enabled. Press Ctrl+C to stop.")
        try:
            # Install pytest-watch if not available
            subprocess.run(["pip", "install", "-q", "pytest-watch"], check=False)
            watch_cmd = ["ptw", "--"] + cmd[3:]  # Remove "python -m pytest"
            subprocess.run(watch_cmd)
        except KeyboardInterrupt:
            print("\n✋ Watch mode stopped")
        return 0
    
    # Run the tests
    exit_code = run_command(cmd, description)
    
    # Print results
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print(f"❌ Tests failed with exit code {exit_code}")
    print(f"{'='*60}\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())