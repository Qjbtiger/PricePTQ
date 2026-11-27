#!/usr/bin/env bash
set -euo pipefail
python -m priceptq.experiments.selection --family transformation "$@"
