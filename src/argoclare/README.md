# Module directory

## Usage

Common functions for ML development are installed as the package [`crusoe`](./src/crusoe/) and functions for plotting in [`crplot`](./src/crplot/). 
From any notebook with crusoe-dev as the conda environment, can run `import crusoe`

## Naming conventions for modules in `crusoe`:

```
import crusoe.argotools as mod_argo  
import crusoe.oceantools as mod_ocean


import crusoe.pcm as mod_pcm 				# clustering
import crusoe.preprocessing as mod_prep		# regression notebooks
import crusoe.regression as mod_reg
import crusoe.crossvalidation as mod_cv
```

## Naming conventions for modules in `crplot`:

```
import crplot.figtools as mod_plot 		# useful plotting fxns
import crplot.figs_* as figs_*           # figs for specific workflow steps

import crplot.southpolarplot as sopo    # for southern ocean plots
```


### Abbreviations:

- pcm: profile classification modeling
- dd: data distributions
