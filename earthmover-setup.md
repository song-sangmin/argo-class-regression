# Setting up Earthmover


## Getting started 

Follow instructions on `https://docs.earthmover.io/quickstart` to get account set up. 
Add "arraylake" as an environment defined in the pixi.toml file. 

Run `pixi install -e arraylake` once. 


Login with: `pixi run -e arraylake arraylake auth login`


## To use with config file: 

Edit the snapshot ID with the dataset you want to use: 
You can print snapshot with
`pixi run -e arraylake python -c "from arraylake import Client; print(Client().get_repo('argo-class-regression/era5-subscription').lookup_branch('main'))"`

## Switching from public to earthmover

Comment out the relevant lines in config file

```
pixi run -e arraylake python scripts/data-processing/create_P1_era5_points.py --config configs/example_drake.toml --datasets core bgc --skip-existing --update-config

pixi run example

```