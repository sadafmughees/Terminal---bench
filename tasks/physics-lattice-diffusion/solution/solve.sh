#!/bin/bash
set -e
cp "$(dirname "$0")/solve_lattice.py" /app/solve_lattice.py
cd /app
python solve_lattice.py --out /app/solution_metrics.json
