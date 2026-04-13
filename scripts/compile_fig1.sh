#!/usr/bin/env bash
# Compile figure1_workflow.tex → figure1_workflow.pdf
# Usage: bash scripts/compile_fig1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
pdflatex -interaction=nonstopmode -halt-on-error figure1_workflow.tex
# Clean auxiliary files
rm -f figure1_workflow.aux figure1_workflow.log
echo "✓ figure1_workflow.pdf generated."
