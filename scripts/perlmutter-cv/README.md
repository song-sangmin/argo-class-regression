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
