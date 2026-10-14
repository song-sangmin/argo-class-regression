#!/bin/bash
#SBATCH --account=m4839
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=modelapp
#SBATCH --output=/pscratch/sd/s/sangsong/dask_cv/logs/%x_%j.out
#
# Train one final clustered model (algorithm + grid per cluster set in CLUSTER_MODELS
# in run_model_application.py) and apply it to coreArgo data on 1 CPU node.
#
# Usage (from /global/homes/s/sangsong/dask-cv-src):
#   sbatch -J app_RFR submit_model_application.sbatch
#   sbatch -J app_mixed -t 02:00:00 submit_model_application.sbatch --run-name crusoe_application_mixed
#
# Any arguments are passed on to run_model_application.py, e.g. also save the fitted models:
#   sbatch -J app_RFR submit_model_application.sbatch --save-models

set -eo pipefail

SRC_DIR=/global/homes/s/sangsong/dask-cv-src

module load python
conda activate dask-cv

# Dask workers import procedures_application, procedures_cv, crossvalidation, regression from here
export PYTHONPATH=$SRC_DIR:$PYTHONPATH
# Parallelism comes from dask workers; keep numeric libraries single-threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd $SRC_DIR
echo "Job $SLURM_JOB_ID on $(hostname): run_model_application.py $*"
echo "Started: $(date)"

python -u run_model_application.py "$@"

echo "Finished: $(date)"
