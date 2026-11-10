# Setting up NERSC skypilot access


- First download the sshproxy executables from this site: https://docs.nersc.gov/connect/mfa/#sshproxy
- Use sshproxy installer
- In Terminal, run `sshproxy -u <username>`
- Open the file `~/.ssh/config` and add the following: 

```
Host perlmutter*.nersc.gov saul*.nersc.gov dtn*.nersc.gov
    User username
    IdentityFile ~/.ssh/nersc
    IdentitiesOnly yes
    ForwardAgent yes
```

- Once this is added, you can connect directly in Terminal using `ssh perlmutter.nersc.gov`. This will connect you to a login node. 


- Open the file `~/.slurm/config` and add the following: 

```
Host perlmutter
    HostName perlmutter.nersc.gov
    User sangsong
    IdentityFile ~/.ssh/nersc
    IdentitiesOnly yes
```

- Open the file `~/.sky/config` and add the following: 

```
slurm:
  allowed_clusters:
    - perlmutter
    # - tillicum
  cluster_configs:
    perlmutter:
      workdir: $SCRATCH/skypilot     # /pscratch/sd/s/sangsong/skypilot
```

# Starting a session

- To login (expires each day), from any Terminal:

```
sshproxy.sh -u sangsong
ssh perlmutter.nersc.gov
```

# Notes on differences from NERSC Jupyterhub setup

- Currently you log into NERSC Jupyterhub, go to `scripts/model-optimization/` and then call `submit_nested_cv.sbatch` from Terminal (login node). 
- Here, Skypilot will generate its own sbatch script, and the commands will run inside the allocation it gets using the information in the .yaml file specified



```
sky launch templates/perlmutter-nestedcv.yaml --cluster cv-xgb \
  --env ALGORITHM=XGB --env GRID=xgb_full --env TAG=cmp20261109
sky down cv-xgb      # when done; Slurm clusters don't autostop
```

- On Perlmutter, you usually don't need to specify a partition (provide only qos and constraint). With Skypilot you need to pick partitions yourself. 

Partition: a named group of nodes, such as compute, ckpt or gpu-a40 on klone.
QOS: a set of rules and limits, such as debug (short, high priority) or regular.
Constraint: a node feature you require, such as cpu or gpu on Perlmutter.

- Note: on Perlmutter using `qos:regular` means you pay for the whole node. For real runs under `regular`, request the whole node, for example cpus: 128 and memory: 400 (leaving headroom below 512).

Under `qos: shared`, you can ask for part of a node, up to half, and you're charged only for that part.


- Location of run logs is different with Skypilot. 

`sky launch` streams output to your laptop terminal. You can press Ctrl-C to stop watching without stopping the job, then reattach later with sky logs <cluster>.
On NERSC: a copy is saved at $SCRATCH/skypilot/.sky_clusters/<cluster>/sky_logs/<job id>/run.log.