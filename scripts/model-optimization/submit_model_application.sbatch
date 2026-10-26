#!/bin/bash
#SBATCH --account=m4839
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=modelapp
#SBATCH --output=logs/%x_%j.out
#
# Train one final clustered model (algorithm + grid per cluster set in CLUSTER_MODELS
# in run_model_application.py, or in a JSON file from compare_algorithms.py) and apply
# it to coreArgo data on 1 CPU node.
#
# Usage (from scripts/model-optimization in the repo, e.g. on NERSC
#   /global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization):
#   sbatch -J app_RFR submit_model_application.sbatch
#   sbatch -J app_mixed -t 02:00:00 submit_model_application.sbatch --run-name crusoe_application_mixed
#
# Any arguments are passed on to run_model_application.py, e.g. also save the fitted models:
#   sbatch -J app_RFR submit_model_application.sbatch --save-models
# or use the best algorithm by cluster from compare_algorithms.py:
#   sbatch -J app_cmp20261015 submit_model_application.sbatch \
#       --cluster-models /pscratch/sd/s/sangsong/argoclare_runs/results/cmp20261015_bestAlgorithms.json

set -eo pipefail

# ==== Machine-specific setup: defaults are for NERSC Perlmutter =======
# On another cluster, nothing here needs editing:
#   Slurm options:  sbatch -A <account> -p <partition> -q <qos> -C <feature> -t ... -o ...
#                   (or SBATCH_ACCOUNT, SBATCH_PARTITION, ... in ~/.bashrc); these override the #SBATCH lines
#   environment:    export ARGOCLARE_ENV_SETUP='<commands that activate the argoclare environment>'
#                   or ARGOCLARE_ENV_SETUP="" to use the environment you submit from
#   config paths:   export ARGOCLARE_MACHINE=<name> (a [machines.<name>] profile in the --config file)
# Submit from scripts/model-optimization (logs go to its logs/ folder), or set ARGOCLARE_SRC_DIR.
SRC_DIR=${ARGOCLARE_SRC_DIR:-$SLURM_SUBMIT_DIR}
if [ ! -f "$SRC_DIR/procedures_application.py" ]; then
    echo "No model-optimization scripts in $SRC_DIR: submit from scripts/model-optimization, or set ARGOCLARE_SRC_DIR"
    exit 1
fi
# Unset: the NERSC commands below. Empty: no setup (use the environment sbatch passed on).
ARGOCLARE_ENV_SETUP=${ARGOCLARE_ENV_SETUP-"module load python && conda activate dask-cv"}
eval "$ARGOCLARE_ENV_SETUP"

# Dask workers import procedures_application, procedures_cv from here (argoclare is pip-installed)
export PYTHONPATH="$SRC_DIR${PYTHONPATH:+:$PYTHONPATH}"
# Parallelism comes from dask workers; keep numeric libraries single-threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Paths in --config <run.toml> come from its [machines.<name>] profile (nersc unless set)
export ARGOCLARE_MACHINE=${ARGOCLARE_MACHINE:-nersc}

cd "$SRC_DIR"
echo "Job $SLURM_JOB_ID on $(hostname): run_model_application.py $*"
echo "Started: $(date)"

python -u run_model_application.py "$@"

echo "Finished: $(date)"
