#!/bin/bash
#
# Submit one nested CV job per algorithm (RFR, ERT, XGB) on identical folds, for
# choosing the best algorithm by cluster with compare_algorithms.py.
# Run from a login node; each algorithm gets its own 1-node job via submit_nested_cv.sbatch.
#
# Usage (from /global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization):
#   ./submit_algorithm_comparison.sh TAG [run_dask_nested_cv.py options]
#   ./submit_algorithm_comparison.sh cmp20261015
#   SEED=7 ./submit_algorithm_comparison.sh cmp20261015_seed7
#
# Run names are crusoe_<ALGORITHM>_<GRID>_<TAG>. Once all jobs finish, on the login node:
#   module load python; conda activate dask-cv
#   python compare_algorithms.py --tag TAG

set -eo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: ./submit_algorithm_comparison.sh TAG [run_dask_nested_cv.py options]"
    exit 1
fi
TAG=$1
shift 1
SEED=${SEED:-42} # same seed for every algorithm, so all runs use identical folds

# ==== Algorithm, tuning grid (tuning_grids.py) and time limit for each job =======
ALGORITHMS=(RFR      ERT      XGB)
GRIDS=(     rfr_full ert_full xgb_full)
TIMES=(     01:00:00 01:00:00 02:00:00)

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

JOB_IDS=()
for i in "${!ALGORITHMS[@]}"; do
    ALGORITHM=${ALGORITHMS[$i]}
    GRID=${GRIDS[$i]}
    RUN_NAME=crusoe_${ALGORITHM}_${GRID}_${TAG}
    JOB_ID=$(sbatch --parsable -J "${ALGORITHM}_${TAG}" -t "${TIMES[$i]}" \
                    "$SCRIPT_DIR/submit_nested_cv.sbatch" "$ALGORITHM" "$GRID" \
                    --run-name "$RUN_NAME" --seed "$SEED" "$@")
    echo "Submitted $ALGORITHM ($GRID) as job $JOB_ID, run name $RUN_NAME"
    JOB_IDS+=("$JOB_ID")
done

ALL_IDS=$(IFS=,; echo "${JOB_IDS[*]}")
echo
echo "Check progress:"
echo "  sacct -j $ALL_IDS --format=JobID,JobName,State,Elapsed"
echo "When all three show COMPLETED, compare on the login node:"
echo "  module load python; conda activate dask-cv"
echo "  python compare_algorithms.py --tag $TAG"
