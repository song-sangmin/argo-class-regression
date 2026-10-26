# Instructions for running on Perlmutter


From terminal: 

```
cd /global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization

sbatch -J RFRcv10x5 submit_nested_cv.sbatch RFR rfr_full
sbatch -J ERTcv10x5 submit_nested_cv.sbatch ERT ert_full
sbatch -J XGBcv10x5 -t 02:00:00 submit_nested_cv.sbatch XGB xgb_full
```

Or debug with shorter version `sbatch -J RFR_test -q debug -t 00:10:00 submit_nested_cv.sbatch RFR rfr_test --n-outer 2 --n-inner 2`

-J names the job and its log file (`logs/<job name>_<job id>.out`, in this folder)
Options before submit_nested_cv.sbatch go to Slurm (-t, -q)
Anything after the grid name goes to the Python script (--run-name, --n-outer, --n-workers and so on). 

Submit from this folder: the job runs from the folder `sbatch` is called from (or `ARGOCLARE_SRC_DIR`). To keep the logs on pscratch as before, add `-o /pscratch/sd/s/sangsong/argoclare_runs/logs/%x_%j.out`.

**Other clusters (e.g. UW Hyak)** need no edits to the job scripts: pass `-A <account> -p <partition> -q <qos> -C <feature>` to `sbatch` (or set `SBATCH_ACCOUNT`, `SBATCH_PARTITION`, ...), set `ARGOCLARE_ENV_SETUP` to the commands that activate the environment (default `module load python && conda activate dask-cv`; `""` to use the environment you submit from), and `ARGOCLARE_MACHINE` to the config's machine profile. See "Running on an HPC system" in `example-workflow.md`.



## Using a run config (optional)

With `--config`, the scripts take `--data`, `--application-data`, `--savepath`, `--sea-ice-clusters` and `--drop-features` from the run config. The paths come from the `[machines.nersc]` profile, and the sbatch scripts set `ARGOCLARE_MACHINE=nersc` for you. Sea ice clusters and dropped features come from the `regression_info_<pcm>.json` that `regression-input-prep.py` writes, so copy it to `x_localdata/` along with the csvs. Options you pass yourself still take precedence. Without `--config`, the defaults at the top of each script apply, as before.

```
C=../../configs/southernOcean_CRUSOE.toml
sbatch -J RFRcv10x5 submit_nested_cv.sbatch RFR rfr_full --config $C
sbatch -J RFR_gmm7 submit_nested_cv.sbatch RFR rfr_full --config $C --n-gmm 7     # another clustering (clustering/README.md)
sbatch -J app_RFR_core submit_model_application.sbatch --config $C
```

The number of clusters is read from the `cluster{k}_prob` columns of the data, so a 7-class fit needs no code changes.

## Check on jobs:

squeue --me                                                # status
tail -f logs/RFR_full_<JOBID>.out                         # live progress (logs/ next to the job scripts)
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

## Running a single algorithm: 
Specify algorithm after .sbatch with its tuning grid, where --run-name saves the file under argoclare-runs/results.
```
sbatch -J RFR_{JOBNAME}  submit_nested_cv.sbatch XGB xgb_full --run-name crusoe_RFR_rfr_full_{JOBNAME} --seed 42 

sbatch -J RFR_noADT  submit_nested_cv.sbatch XGB xgb_full --run-name crusoe_RFR_rfr_full_noADT --seed 42 --drop-features adt
```

Apply model to Core Argo data: (optional --cluster-models, by default RFR for all)
```
sbatch -J app_RFR_core submit_model_application.sbatch

sbatch -J app_cmp20261015 submit_model_application.sbatch \
    --cluster-models /pscratch/sd/s/sangsong/argoclare_runs/results/cmp20261015_bestAlgorithms.json
```


## Synthetic data: train and apply with the real-world algorithms

No algorithm comparison here. Reuse the `bestAlgorithms.json` from the real-world comparison and point the application job at the synthetic files. `regression-input-prep.py` prints the sea ice clusters and features to use for each dataset (synthetic: cluster 6 only, no `adt`).

With the synthetic run config, the synthetic files, sea ice cluster 6 and `--drop-features adt` are all set:
```
sbatch -J app_synthetic submit_model_application.sbatch --config ../../configs/synthetic_CRUSOE.toml \
    --cluster-models /pscratch/sd/s/sangsong/argoclare_runs/results/cmp20261015_bestAlgorithms.json \
    --run-name crusoe_synthetic_application_cmp20261017
```

Without a config:

Cluster-dependent algorithm (using --cluster-models)
```
X=/global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization/x_localdata
sbatch -J app_synthetic submit_model_application.sbatch \
    --cluster-models /pscratch/sd/s/sangsong/argoclare_runs/results/cmp20261015_bestAlgorithms.json \
    --data $X/combined_trainval_synthetic_pc8_gmm6_dbar500_seaiceclusters6_acc20261017.csv \
    --application-data $X/coreArgo_application_synthetic_pc8_gmm6_dbar500_acc20261017.csv \
    --sea-ice-clusters 6 --drop-features adt \
    --run-name crusoe_synthetic_application_cmp20261017
```

```
X=/global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization/x_localdata
sbatch -J app_synthetic submit_model_application.sbatch \
    --cluster-models /pscratch/sd/s/sangsong/argoclare_runs/results/cmp20261015_bestAlgorithms.json \
    --data $X/combined_trainval_synthetic_pc8_gmm6_dbar500_seaiceclusters6_acc20261017.csv \
    --application-data $X/coreArgo_application_synthetic_pc8_gmm6_dbar500_acc20261017.csv \
    --sea-ice-clusters 6 --drop-features adt \
    --run-name crusoe_synthetic_application_cmp20261017
```

`--sea-ice-clusters` and `--drop-features` also work with `submit_nested_cv.sbatch` (passed on to `run_dask_nested_cv.py`), e.g. to get synthetic CV errors for one algorithm.


=========================


## Choosing the best algorithm by cluster (archive)

Three steps, all started from the login node in `/global/cfs/cdirs/m4839/sangsong/github-repos/argo-class-regression/scripts/model-optimization`. Pick a tag to name the comparison, e.g. `cmp20261015`.

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
    --cluster-models /pscratch/sd/s/sangsong/argoclare_runs/results/cmp20261015_bestAlgorithms.json
```

Without `--cluster-models`, `run_model_application.py` uses the `CLUSTER_MODELS` dict at the top of the script. The run description records which one was used. Add `--save-models` to also pickle the fitted models.
