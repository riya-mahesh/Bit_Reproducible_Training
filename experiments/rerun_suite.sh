#!/usr/bin/env bash
# Runs only the operation suite for one configuration, on every GPU of this machine or GPUS.
#   experiments/rerun_suite.sh cuda124
set -uo pipefail
source "$(dirname "$0")/common.sh"
load_config "$1"
"$REPO"/gpt2_ops_suite/run_suite.sh
