#!/bin/bash
#SBATCH --account=m4839
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=nestedcv
#SBATCH --output=logs/%x_%j.out
#
# Nested CV hyperparameter tuning for one algorithm + tuning grid on 1 CPU node.
#
# Usage (from scripts/model-optimization in the repo, e.g. on NERSC
#   /global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization):
#   sbatch -J RFR_full submit_nested_cv.sbatch RFR rfr_full
#   sbatch -J XGB_full -t 02:00:00 submit_nested_cv.sbatch XGB xgb_full
#
# Anything after the grid name is passed on to run_dask_nested_cv.py, e.g. a quick test:
#   sbatch -J RFR_test -q debug -t 00:10:00 submit_nested_cv.sbatch RFR rfr_test --n-outer 2 --n-inner 2

set -eo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: sbatch [sbatch options] submit_nested_cv.sbatch ALGORITHM GRID [run_dask_nested_cv.py options]"
    exit 1
fi
ALGORITHM=$1
GRID=$2
shift 2

# ==== Machine-specific setup: defaults are for NERSC Perlmutter =======
# On another cluster, nothing here needs editing:
#   Slurm options:  sbatch -A <account> -p <partition> -q <qos> -C <feature> -t ... -o ...
#                   (or SBATCH_ACCOUNT, SBATCH_PARTITION, ... in ~/.bashrc); these override the #SBATCH lines
#   environment:    export ARGOCLARE_ENV_SETUP='<commands that activate the argoclare environment>'
#                   or ARGOCLARE_ENV_SETUP="" to use the environment you submit from
#   config paths:   export ARGOCLARE_MACHINE=<name> (a [machines.<name>] profile in the --config file)
# Submit from scripts/model-optimization (logs go to its logs/ folder), or set ARGOCLARE_SRC_DIR.
SRC_DIR=${ARGOCLARE_SRC_DIR:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SRC_DIR/procedures_cv.py" ]; then
    echo "No model-optimization scripts in $SRC_DIR: submit from scripts/model-optimization, or set ARGOCLARE_SRC_DIR"
    exit 1
fi
# Unset: the NERSC commands below. Empty: no setup (use the environment sbatch passed on).
ARGOCLARE_ENV_SETUP=${ARGOCLARE_ENV_SETUP-"module load python && conda activate dask-cv"}
eval "$ARGOCLARE_ENV_SETUP"

# Dask workers import procedures_cv from here (argoclare is pip-installed)
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Parallelism comes from dask workers; keep numeric libraries single-threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Paths in --config <run.toml> come from its [machines.<name>] profile (nersc unless set)
export ARGOCLARE_MACHINE=${ARGOCLARE_MACHINE:-nersc}

cd "$SRC_DIR"
echo "Job $SLURM_JOB_ID on $(hostname): $ALGORITHM $GRID $*"
echo "Started: $(date)"

python -u run_dask_nested_cv.py "$ALGORITHM" "$GRID" "$@"

echo "Finished: $(date)"
       