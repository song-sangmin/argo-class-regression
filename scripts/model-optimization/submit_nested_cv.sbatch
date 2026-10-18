#!/bin/bash
#SBATCH --account=m4839
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=nestedcv
#SBATCH --output=/pscratch/sd/s/sangsong/argoclare_runs/logs/%x_%j.out
#
# Nested CV hyperparameter tuning for one algorithm + tuning grid on 1 CPU node.
#
# Usage (from /global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization):
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

SRC_DIR=/global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization

module load python
conda activate dask-cv

# Dask workers import procedures_cv from here (argoclare is pip-installed)
export PYTHONPATH=$SRC_DIR:$PYTHONPATH
# Parallelism comes from dask workers; keep numeric libraries single-threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $SRC_DIR
echo "Job $SLURM_JOB_ID on $(hostname): $ALGORITHM $GRID $*"
echo "Started: $(date)"

python -u run_dask_nested_cv.py "$ALGORITHM" "$GRID" "$@"

echo "Finished: $(date)"
       