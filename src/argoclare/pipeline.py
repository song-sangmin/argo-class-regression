"""Run configuration shared by the pipeline scripts (data-processing, clustering, regression,
model-optimization).

Every script takes --config <file.toml> (see configs/ at the repo root). A config holds a run's
settings (region, years, QC flags, clustering, regression, ...) and, in [files], where each
stage reads and writes. File paths can use these placeholders:
    {data_root} {run_dir}   from [run]
    {dataset}               core, bgc, soccom, socat, ...
    {yr}                    for files with one year each
    {pcm}                   the clustering tag, e.g. pc8_gmm6_dbar500 (from [P2], see pcm_tag())
Paths can also use ~ and environment variables, e.g. "$PSCRATCH/argoclare-data" on NERSC.
A file key can be given per dataset (e.g. L1_core), which is used before the shared key (L1),
and per clustering in a [files.<pcm>] table, which is used before both. Files left out of
[files] use the run folder layout in DEFAULT_FILES:

    <data_root>/<name>/            run_dir
        config.toml                copy of the config, made by the first script run
        provenance/                one json per script run (command, git commit, versions)
        L0/ L1-qc/ L2-mask/ L3-interp/ socat/ ancillary/ P1/
        P2/<pcm>/                  clustering fits and classified samples, one folder per fit
        regression/<pcm>/          regression inputs
        model-optimization/<pcm>/  nested CV and application results

Machines: a [machines.<name>] table in the config overrides any setting (e.g. data_root,
run_dir, [files] paths, n_jobs) on that machine. It is chosen with --machine <name> or the
ARGOCLARE_MACHINE environment variable, so one config works on a laptop and on NERSC.

Safeguards:
    - outputs that already exist are not overwritten unless --overwrite is passed
    - a config that differs from <run_dir>/config.toml is refused unless --update-config is
      passed (the old copy is kept as config_<time>.toml)
    - --out-dir writes this script's outputs to another folder (same file names), for tests

In a notebook:
    from argoclare import pipeline
    run = pipeline.load('configs/southernOcean_CRUSOE.toml')
    run.status()                                  # every [files] entry and whether it exists
    coreINDEX = run.open('L3_index', 'core')
    p2 = run.open('P2_trainval', pcm='pc8_gmm6_dbar500')
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from importlib import metadata

try:
    import tomllib
except ModuleNotFoundError: # Python < 3.11
    import tomli as tomllib

DEFAULT_FILES = {
    'L0_profiles': '{run_dir}/L0/{dataset}_yr{yr}_profiles.nc',
    'L0_index': '{run_dir}/L0/{dataset}_yr{yr}_index.csv',
    'L1': '{run_dir}/L1-qc/{dataset}_QC_yr{yr}.csv',
    'L2': '{run_dir}/L2-mask/{dataset}_open_ocean_INDEX_{yr}.nc',
    'L3_data': '{run_dir}/L3-interp/{dataset}DATA_byProfid.nc',
    'L3_index': '{run_dir}/L3-interp/{dataset}INDEX_byProfid.nc',
    'socat_L2': '{run_dir}/socat/socat_resampled_open_ocean_INDEX.nc',
    'mbl': '{run_dir}/ancillary/co2_mbl_combined.nc',
    'P1': '{run_dir}/P1/P1_{dataset}_features.nc',
    'P2_finder': '{run_dir}/P2/{pcm}/PCM_finder_{pcm}.parquet',
    'P2_fit': '{run_dir}/P2/{pcm}/PCM_fit_{pcm}.json',
    'P2_model': '{run_dir}/P2/{pcm}/PCM_model_{pcm}.joblib',
    'P2': '{run_dir}/P2/{pcm}/P2_{dataset}_{pcm}.parquet',
    'P2_trainval': '{run_dir}/P2/{pcm}/P2_combined_trainval_{pcm}.parquet',
    'P2_test': '{run_dir}/P2/{pcm}/P2_combined_test_{pcm}.parquet',
    'P2_application': '{run_dir}/P2/{pcm}/P2_application_{pcm}.parquet',
    'reg_trainval': '{run_dir}/regression/{pcm}/combined_trainval_{pcm}.csv',
    'reg_test': '{run_dir}/regression/{pcm}/combined_test_{pcm}.csv',
    'reg_application': '{run_dir}/regression/{pcm}/application_{pcm}.csv',
    'reg_info': '{run_dir}/regression/{pcm}/regression_info_{pcm}.json',
    'model_results': '{run_dir}/model-optimization/{pcm}',
    'argopy_cache': '{run_dir}/argopy-cache',
}

PACKAGES = ['argoclare', 'argopy', 'xarray', 'pandas', 'numpy', 'scikit-learn', 'gsw', 'PyCO2SYS']


def pcm_tag(n_pca, n_gmm, dbar_limit, exclude_years=None):
    """Clustering tag used in [files] paths: pc<n_pca>_gmm<n_gmm>_dbar<dbar_limit>[_excl<years>]."""
    tag = f'pc{n_pca}_gmm{n_gmm}_dbar{dbar_limit:g}'
    if exclude_years:
        tag += '_excl' + '-'.join(str(y) for y in sorted(exclude_years))
    return tag


def parser(doc):
    """argparse parser with the options every script shares."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='Run config (.toml), e.g. configs/example.toml')
    p.add_argument('--machine', default=None,
                   help='[machines.<name>] profile in the config (default: $ARGOCLARE_MACHINE)')
    p.add_argument('--overwrite', action='store_true', help='Replace output files that already exist')
    p.add_argument('--update-config', action='store_true',
                   help='Accept a config that differs from the copy in the run folder')
    p.add_argument('--out-dir', default=None,
                   help='Write outputs to this folder instead of the configured paths (for tests)')
    return p


def add_pcm_args(p):
    """Clustering options that override [P2] (a different fit gets its own {pcm} folder)."""
    p.add_argument('--n-pca', type=int, default=None, help='PCA components (default: [P2] n_pca)')
    p.add_argument('--n-gmm', type=int, default=None, help='GMM classes (default: [P2] n_gmm)')
    p.add_argument('--dbar-limit', type=float, default=None, help='Fit pressure levels < this (default: [P2] dbar_limit)')
    p.add_argument('--exclude-years', type=int, nargs='+', default=None,
                   help='Years left out of the fit, then classified with it (default: [P2] exclude_years)')
    return p


def start(args, script):
    """Load the config named in args, check it against the run folder copy, and return a Run."""
    return Run(args.config, script, args=args)


def load(config_path, machine=None, **pcm):
    """A Run for reading paths and files (notebooks, model-optimization); writes nothing.
    pcm: n_pca, n_gmm, dbar_limit, exclude_years overrides, as on the command line."""
    args = argparse.Namespace(machine=machine, **pcm)
    return Run(config_path, 'notebook', args=args, freeze=False)


def add_model_args(p):
    """--config, --machine and the clustering options for the model-optimization scripts."""
    p.add_argument('--config', default=None,
                   help='Run config (.toml): sets the default --data, --application-data, --savepath, '
                        '--sea-ice-clusters and --drop-features (with --machine or $ARGOCLARE_MACHINE)')
    p.add_argument('--machine', default=None, help='[machines.<name>] profile in the config')
    return add_pcm_args(p)


def fill_model_args(args):
    """Fill the model-optimization options left unset from the run config ([files] reg_trainval,
    reg_application, reg_test (if it exists), model_results, [regression] cv_type, n_outer, n_inner, n_folds, and the sea ice clusters, features and target in reg_info:
    args.features, args.target). Returns the Run, or None without --config."""
    if not getattr(args, 'config', None):
        return None
    run = load(args.config, machine=args.machine, n_pca=args.n_pca, n_gmm=args.n_gmm,
               dbar_limit=args.dbar_limit, exclude_years=args.exclude_years)
    if hasattr(args, 'data') and args.data is None:
        args.data = run.input('reg_trainval')
    if hasattr(args, 'application_data') and args.application_data is None:
        args.application_data = run.input('reg_application')
    if hasattr(args, 'test_data') and args.test_data is None and not getattr(args, 'no_test', False):
        test_file = run.path('reg_test')  # test samples from regression-input-prep.py, if it made them
        args.test_data = test_file if os.path.exists(test_file) else None
    if args.savepath is None:
        args.savepath = run.path('model_results')
    if getattr(args, 'cv_type', 0) is None:
        args.cv_type = run.section('regression').get('cv_type')
    for key in ('n_outer', 'n_inner', 'n_folds'): # fold counts, e.g. [regression] n_outer = 5
        if getattr(args, key, 0) is None:
            setattr(args, key, run.section('regression').get(key))
    info_file = run.path('reg_info')
    if os.path.exists(info_file):
        with open(info_file) as f:
            info = json.load(f)
        if hasattr(args, 'data') and info.get('trainval_file') != os.path.basename(args.data):
            print(f'Note: {os.path.basename(info_file)} is for {info.get("trainval_file")}, not '
                  f'{os.path.basename(args.data)}; not using its sea ice clusters or features')
        else:
            if getattr(args, 'sea_ice_clusters', 0) is None:
                args.sea_ice_clusters = info['sea_ice_clusters']
            if hasattr(args, 'drop_features') and not args.drop_features:
                args.drop_features = info.get('drop_features', [])
            args.features, args.target = info.get('features'), info.get('target')
    elif hasattr(args, 'data'):
        if hasattr(args, 'drop_features') and not args.drop_features:
            args.drop_features = run.section('regression').get('drop_features', [])
        if args.sea_ice_clusters is None:
            print(f'Note: no {os.path.basename(info_file)}; pass --sea-ice-clusters, or the script default is used')
    print(f'Run config {os.path.basename(args.config)} (pcm {run.pcm}, machine {run.machine or "default"})')
    return run


def _expand(path):
    """~ and environment variables (e.g. $PSCRATCH on NERSC) in a config path."""
    expanded = os.path.expandvars(os.path.expanduser(path))
    if '$' in expanded:
        raise SystemExit(f'Config path {path!r} uses an environment variable that is not set here. '
                         'Set it, or choose another [machines] profile (ARGOCLARE_MACHINE / --machine).')
    return expanded


def _merge(base, over):
    """Recursive dict update (for machine profiles)."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Run:
    def __init__(self, config_path, script, args=None, freeze=True):
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, 'rb') as f:
            self.raw_cfg = tomllib.load(f)
        self.script = os.path.basename(script)
        self.args = args
        self.overwrite = getattr(args, 'overwrite', False)
        self.out_dir = getattr(args, 'out_dir', None)
        self.inputs, self.outputs = [], []
        self.started = datetime.now()

        # Machine profile: [machines.<name>] overrides the rest of the config
        self.machine = getattr(args, 'machine', None) or os.environ.get('ARGOCLARE_MACHINE')
        machines = self.raw_cfg.get('machines', {})
        self.cfg = {k: v for k, v in self.raw_cfg.items() if k != 'machines'}
        if self.machine in machines:
            self.cfg = _merge(self.cfg, machines[self.machine])
        elif self.machine:
            print(f'Note: no [machines.{self.machine}] in {os.path.basename(config_path)}; using the defaults')

        run = self.cfg['run']
        self.name = run['name']
        # Relative paths are relative to the config file, so a run does not depend on the cwd
        self.config_dir = os.path.dirname(self.config_path)
        self.data_root = os.path.normpath(os.path.join(self.config_dir, _expand(run['data_root'])))
        self.run_dir = os.path.normpath(os.path.join(self.config_dir, _expand(
            run.get('run_dir', '{data_root}/{name}').format(data_root=self.data_root, name=self.name))))
        self.ref_time = run.get('ref_time', '2014-01-01')
        if freeze:
            self._freeze(getattr(args, 'update_config', False))

    # %% Settings

    def section(self, name):
        """A config table, e.g. run.section('L1') or run.section('P1.soccom'); {} if missing."""
        table = self.cfg
        for key in name.split('.'):
            table = table.get(key, {})
        return table

    def years(self, stage=None):
        """--years if given, else the stage's own years (e.g. [P1.core] years), else [run] years."""
        if getattr(self.args, 'years', None):
            return self.args.years
        return list(self.section(stage).get('years', self.cfg['run']['years']) if stage
                    else self.cfg['run']['years'])

    @property
    def box(self):
        """[lon_min, lon_max, lat_min, lat_max]"""
        return list(self.cfg['region']['box'])

    @property
    def pres(self):
        """[pres_min, pres_max] in dbar"""
        return list(self.cfg['region'].get('pres', [0, 2000]))

    def in_box(self, lon, lat):
        """Boolean mask of positions inside the region box (longitudes in either -180..180 or 0..360)."""
        lon_min, lon_max, lat_min, lat_max = self.box
        lon180 = (lon + 180) % 360 - 180
        lon_ok = (lon180 >= lon_min) & (lon180 <= lon_max) if lon_max - lon_min < 360 else True
        return lon_ok & (lat >= lat_min) & (lat < lat_max)

    @property
    def p2(self):
        """[P2] settings with the command line overrides (--n-pca, --n-gmm, --dbar-limit, --exclude-years)."""
        p2 = dict(self.section('P2'))
        for key in ['n_pca', 'n_gmm', 'dbar_limit', 'exclude_years']:
            if getattr(self.args, key, None) is not None:
                p2[key] = getattr(self.args, key)
        return p2

    @property
    def pcm(self):
        """Clustering tag for {pcm}, e.g. pc8_gmm6_dbar500; None without a [P2] table."""
        p2 = self.p2
        if not all(k in p2 for k in ['n_pca', 'n_gmm', 'dbar_limit']):
            return None
        return pcm_tag(p2['n_pca'], p2['n_gmm'], p2['dbar_limit'], p2.get('exclude_years'))

    # %% Files

    def _template(self, key, dataset=None, pcm=None):
        files = self.cfg.get('files', {})
        keys = ([f'{key}_{dataset}'] if dataset else []) + [key]
        tables = ([files[pcm]] if pcm and isinstance(files.get(pcm), dict) else []) + [files, DEFAULT_FILES]
        for table in tables:
            for k in keys:
                if k in table and not isinstance(table[k], dict):
                    return table[k]
        raise SystemExit(f'{self.config_path}: no [files] entry for "{key}"'
                         + (f' (or "{key}_{dataset}")' if dataset else ''))

    def path(self, key, dataset=None, pcm=None, **fmt):
        """Resolved path (or list of paths) for a [files] key, with no existence checks."""
        pcm = pcm or self.pcm
        template = self._template(key, dataset, pcm)
        def resolve(t):
            if '{pcm}' in t and pcm is None:
                raise SystemExit(f'{self.config_path}: [files] {key} uses {{pcm}}, but [P2] has no n_pca, n_gmm, dbar_limit')
            p = _expand(t.format(data_root=self.data_root, run_dir=self.run_dir,
                                 dataset=dataset, pcm=pcm, **fmt))
            return os.path.normpath(os.path.join(self.config_dir, p))
        return [resolve(t) for t in template] if isinstance(template, list) else resolve(template)

    def template(self, key, dataset=None):
        """Path with {yr} left in, for files opened one year at a time (e.g. ERA5)."""
        return self.path(key, dataset, yr='{yr}')

    def input(self, key, dataset=None, **fmt):
        """Path of an input file; stops with the config key if it does not exist."""
        paths = self.path(key, dataset, **fmt)
        for p in paths if isinstance(paths, list) else [paths]:
            if not os.path.exists(p):
                raise SystemExit(f'Missing input for [files] {key} ({dataset or "all"}): {p}')
            self.inputs.append(p)
        return paths

    def p1_features(self):
        """Columns P1 should compute: [P1] features, plus the [regression] features (without
        drop_features) and target. None if neither table lists features (compute all)."""
        p1, reg = self.section('P1'), self.section('regression')
        if 'features' not in p1 and 'features' not in reg:
            return None
        drop = reg.get('drop_features', [])
        wanted = p1.get('features', []) + [f for f in reg.get('features', []) if f not in drop]
        return list(dict.fromkeys(wanted + ([reg['target']] if 'target' in reg else [])))

    def feature_files(self, years, names=('slp', 'wind', 'adt', 'seaice', 'mbl')):
        """Gridded P1 feature files for p1_features.add_all_features(), as paths with {yr};
        only the given names (p1_features.needed_files()), each checked for every year first."""
        keys = {'slp': 'era5_slp', 'wind': 'era5_wind', 'adt': 'adt', 'seaice': 'seaice'}
        files = {}
        for name, key in keys.items():
            if name in names:
                for yr in years:
                    self.input(key, yr=yr)
                files[name] = self.template(key)
        if 'mbl' in names:
            files['mbl'] = self.input('mbl')
        return files

    def output(self, key, dataset=None, **fmt):
        """Path to write an output to; stops if it exists (unless --overwrite) and makes its folder.
        With --out-dir, the file keeps its name but goes to that folder."""
        p = self.path(key, dataset, **fmt)
        if self.out_dir:
            p = os.path.join(os.path.abspath(self.out_dir), os.path.basename(p))
        if os.path.exists(p) and not self.overwrite:
            raise SystemExit(f'Output exists, pass --overwrite to replace it: {p}')
        os.makedirs(os.path.dirname(p), exist_ok=True)
        self.outputs.append(p)
        return p

    def open(self, key, dataset=None, **fmt):
        """Open a file: netcdf with xarray, parquet/csv with pandas, json as a dict."""
        import pandas as pd
        import xarray as xr
        path = self.path(key, dataset, **fmt)
        if path.endswith('.nc'):
            return xr.open_dataset(path)
        if path.endswith('.parquet'):
            return pd.read_parquet(path)
        if path.endswith('.json'):
            with open(path) as f:
                return json.load(f)
        return pd.read_csv(path, index_col=0)

    def status(self, years=None):
        """Print every [files] entry (with the current {pcm}) and whether it exists."""
        files = self.cfg.get('files', {})
        years = years or self.years()
        print(f'{self.name} ({self.config_path}), pcm {self.pcm}, machine {self.machine or "default"}')
        print(f'  data_root {self.data_root}\n  run_dir   {self.run_dir}')
        keys = {k for k, v in files.items() if not isinstance(v, dict)}
        if isinstance(files.get(self.pcm), dict):
            keys |= set(files[self.pcm])
        for key in sorted(keys):
            base, _, ds = key.rpartition('_') if key.endswith(('_core', '_bgc', '_soccom', '_socat')) else (key, '', None)
            try:
                paths = self.path(base, ds, yr='{yr}')
            except SystemExit:
                continue
            paths = paths if isinstance(paths, list) else [paths]
            missing = []
            for p in paths:
                if '{yr}' in p:
                    missing += [yr for yr in years if not os.path.exists(p.format(yr=yr))]
                elif not os.path.exists(p):
                    missing.append(os.path.basename(p))
            print(f'  {key:18s} {"ok" if not missing else "MISSING " + str(missing)[:60]:20s} {os.path.basename(paths[0])}')

    # %% Provenance

    def _freeze(self, update):
        """Copy the config into the run folder, or check it matches the copy already there."""
        frozen = os.path.join(self.run_dir, 'config.toml')
        if os.path.exists(frozen):
            if os.path.samefile(frozen, self.config_path):
                return
            with open(frozen, 'rb') as f:
                same = tomllib.load(f) == self.raw_cfg
            if same:
                return
            if not update:
                raise SystemExit(f'{self.config_path} differs from the config this run was started with '
                                 f'({frozen}). Use a new run name, or pass --update-config.')
            shutil.copy(frozen, os.path.join(self.run_dir, f'config_{self.started:%Y%m%d_%H%M%S}.toml'))
        os.makedirs(self.run_dir, exist_ok=True)
        shutil.copy(self.config_path, frozen)

    def finish(self, **extra):
        """Write provenance/<script>_<time>.json for this script run; returns its path."""
        def git(*cmd):
            try:
                return subprocess.run(['git', *cmd], cwd=os.path.dirname(os.path.abspath(__file__)),
                                      capture_output=True, text=True, check=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return None
        versions = {}
        for pkg in PACKAGES:
            try:
                versions[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                pass
        record = {'script': self.script,
                  'command': ' '.join([os.path.basename(sys.executable)] + sys.argv),
                  'config': self.config_path,
                  'machine': self.machine,
                  'pcm': self.pcm,
                  'args': {k: v for k, v in vars(self.args).items()} if self.args else {},
                  'started': self.started.isoformat(timespec='seconds'),
                  'finished': datetime.now().isoformat(timespec='seconds'),
                  'git_commit': git('rev-parse', 'HEAD'),
                  'git_dirty': bool(git('status', '--porcelain', '--', '.')),
                  'python': sys.version.split()[0],
                  'packages': versions,
                  'inputs': sorted(set(self.inputs)),
                  'outputs': self.outputs,
                  **extra}
        folder = os.path.join(self.run_dir, 'provenance')
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'{os.path.splitext(self.script)[0]}_{self.started:%Y%m%d_%H%M%S}.json')
        with open(path, 'w') as f:
            json.dump(record, f, indent=2, default=str)
        print('Provenance: ' + path)
        return path

    def attrs(self):
        """netcdf attributes that tie a file to its run."""
        return {'run_name': self.name, 'run_config': os.path.join(self.run_dir, 'config.toml'),
                'script': self.script}
