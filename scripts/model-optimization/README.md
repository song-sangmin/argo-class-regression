# Instructions for running on Perlmutter


From terminal: 

```
cd /global/homes/s/sangsong/dask-cv-src

sbatch -J RFRcv10x5 submit_nested_cv.sbatch RFR rfr_full
sbatch -J ERTcv10x5 submit_nested_cv.sbatch ERT ert_full
sbatch -J XGBcv10x5 -t 02:00:00 submit_nested_cv.sbatch XGB xgb_full
```

Or debug with shorter version `sbatch -J RFR_test -q debug -t 00:10:00 submit_nested_cv.sbatch RFR rfr_test --n-outer 2 --n-inner 2`

-J names the job and its log file
Options before submit_nested_cv.sbatch go to Slurm (-t, -q)
Anything after the grid name goes to the Python script (--run-name, --n-outer, --n-workers and so on). 



## Check on jobs:

squeue --me                                                # status
tail -f /pscratch/sd/s/sangsong/dask_cv/logs/RFR_full_<JOBID>.out   # live progress
scancel <JOBID>                                            # cancel
sacct -j <JOBID> --format=JobID,State,Elapsed              # after it finishes


## Time allocation

A batch job lasts only as long as the submit_nested_cv.sbatch script runs. Once the last line (echo "Finished: ...") runs, the job exits and Slurm frees the node and stops charging hours.

    `run_dask_nested_cv` saves the results.
    The with LocalCluster(...) block in `run_dask_nested_cv.py` closes, which shuts down the Dask workers.
    Python exits, the script finishes, and the job is marked COMPLETED.

`salloc` holds the node until you exit or hit the time limit ==> Ctrl+C or scancel 

## Cancelling a job early 

When you'd still use `scancel`:

To stop a job early, for example if you submitted the wrong grid.
If a job hangs. If Python gets stuck (say, a Dask worker stalls), the job keeps running until the --time limit and Slurm kills it then. You're charged for that time, so if the log stops changing for a long time, cancel it yourself.
If Python crashes, `set -eo pipefail` makes the script stop right away, so a failed run also ends the job without waiting for the time limit. It shows as FAILED in sacct instead of COMPLETED.


## Choosing the best algorithm by cluster

Three steps, all started from the login node in `/global/homes/s/sangsong/dask-cv-src`. Pick a tag to name the comparison, e.g. `cmp20261015`.

**1. Submit nested CV for RFR, ERT and XGB** (three 1-node jobs that run in parallel):

```
./submit_algorithm_comparison.sh cmp20261015
```

All three runs use the same `--seed` (42 by default, or `SEED=7 ./submit_algorithm_comparison.sh ...`), so they are tested on identical folds. Run names are `crusoe_<ALGORITHM>_<GRID>_<TAG>`. The grid and time limit for each algorithm are set at the top of the script. The script prints the job IDs and the `sacct` command to check on them.

**2. Compare on the login node** once all three jobs show COMPLETED (light pandas work, no node charge):

```
module load python
conda activate dask-cv
python compare_algorithms.py --tag cmp20261015
```

For each cluster k this scores that cluster's own model on its own rows (`cluster{k}_pred` vs `delta_pco2` where `cluster == k`) and picks the algorithm with the lowest RMSE. Calibrated errors (`cluster{k}_calpred`) are stored alongside but not used to choose, since the nested CV calibration is fitted on the same validation rows. A warning is printed for any cluster where the calibrated RMSE would pick a different algorithm. It stops with an error if any of the three runs has no output file.

Saved to the results folder:
- `<tag>_algorithmComparison.csv`: RMSE, MAE, bias and n for every cluster x algorithm (raw and `_cal`)
- `<tag>_overallComparison.csv`: overall `weighted_pred` errors for each run, plus the combined best choice
- `<tag>_bestAlgorithms.json`: `{cluster: [algorithm, grid]}` for step 3

**3. Train and apply the final model** with the chosen algorithms:

```
sbatch -J app_cmp20261015 submit_model_application.sbatch \
    --cluster-models /pscratch/sd/s/sangsong/dask_cv/results/cmp20261015_bestAlgorithms.json
```

Without `--cluster-models`, `run_model_application.py` uses the `CLUSTER_MODELS` dict at the top of the script. The run description records which one was used. Add `--save-models` to also pickle the fitted models.
