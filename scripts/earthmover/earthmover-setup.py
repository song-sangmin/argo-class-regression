import xarray as xr
from arraylake import Client

ORG_NAME = 'argo-class-regression'


def main():
    client = Client()
    repo = client.get_repo(f'{ORG_NAME}/era5-subscription')
    session = repo.readonly_session(branch='main')
    ds = xr.open_zarr(session.store, chunks=None, group='single/spatial')

    # storage = icechunk.s3_storage(
    #     bucket="earthmover-icechunk-era5", prefix="icechunkV2",
    #     region="us-east-1", anonymous=True,
    #     endpoint_url="https://s3.us-east-1.amazonaws.com",  # only if an AWS profile sets another
    # )
    # session = icechunk.Repository.open(storage).readonly_session("main")
    # ds = xr.open_zarr(session.store, group="single/temporal", zarr_format=3, chunks=None)
    # ds.tp.sel(latitude=63.25, longitude=208.75, valid_time="2023-09-13")  # lon 0–360, m

    # for g in ["single/spatial", "single/temporal"]:
    #     ds = xr.open_zarr(session.store, group=g, chunks=None)
    #     print(g, ds["u10"].encoding["chunks"], ds.sizes)

    print(list(ds.data_vars))
    # print(ds)


if __name__ == '__main__':
    main()
