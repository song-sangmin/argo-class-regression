"""Run configuration shared by the data-processing scripts (L0 through P1).

Every create_*.py script takes --config <file.toml> (see configs/ at the repo root). A config
holds a run's settings (region, years, QC flags, ...) and, in [files], where each stage reads
and writes. File paths can use {data_root}, {run_dir}, {dataset} (core or bgc) and {yr}. A
file key can be given per dataset (e.g. L1_core), which is used before the shared key (L1);
outputs left out of [files] use the run folder layout in DEFAULT_FILES:

    <data_root>/<name>/            run_dir
        config.toml                copy of the config, made by the first script run
        provenance/                one json per script run (command, git commit, versions)
        L0/ L1-qc/ L2-mask/ L3-interp/ socat/ ancillary/ P1/

Safeguards:
    - outputs that already exist are not overwritten unless --overwrite is passed
    - a config that differs from <run_dir>/config.toml is refused unless --update-config is
      passed (the old copy is kept as config_<time>.toml)
    - --out-dir writes this script's outputs to another folder (same file names), for tests
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from datetime import datetime
from importlib import metadata

DEFAULT_FILES = {
    'L0_profiles': '{run_dir}/L0/{dataset}_yr{yr}_profiles.nc',
    'L0_index': '{run_dir}/L0/{dataset}_yr{yr}_index.csv',
    'L1': '{run_dir}/L1-qc/{dataset}_QC_yr{yr}.csv',
    'L2': '{run_dir}/L2-mask/{dataset}_open_ocean_INDEX_{yr}.nc',
    'L3_data': '{run_dir}/L3-interp/{dataset}DATA_byProfid.nc',
    'L3_index': '{run_dir}/L3-interp/{dataset}INDEX_byProfid.nc',
    'socat_L2': '{run_dir}/socat/socat_resampled_open_ocean_INDEX.nc',
    'mbl': '{run_dir}/ancillary/co2_mbl_combined.nc',
    'P1_core': '{run_dir}/P1/P1_coreArgo_application_features.nc',
    'P1_bgc': '{run_dir}/P1/P1_bgcArgo_features.nc',
    'P1_socat': '{run_dir}/P1/P1_socat_features.nc',
    'argopy_cache': '{run_dir}/argopy-cache',
}

PACKAGES = ['argoclare', 'argopy', 'xarray', 'pandas', 'numpy', 'gsw', 'PyCO2SYS']


def parser(doc):
    """argparse parser with the options every script shares."""
    p = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', required=True, help='Run config (.toml), e.g. configs/example.toml')
    p.add_argument('--overwrite', action='store_true', help='Replace output files that already exist')
    p.add_argument('--update-config', action='store_true',
                   help='Accept a config that differs from the copy in the run folder')
    p.add_argument('--out-dir', default=None,
                   help='Write outputs to this folder instead of the configured paths (for tests)')
    return p


def start(args, script):
    """Load the config named in args, check it against the run folder copy, and return a Run."""
    return Run(args.config, script, args=args)


class Run:
    def __init__(self, config_path, script, args=None):
        self.config_path = os.path.abspath(config_path)
        with open(self.config_path, 'rb') as f:
            self.cfg = tomllib.load(f)
        self.script = os.path.basename(script)
        self.args = args
        self.overwrite = getattr(args, 'overwrite', False)
        self.out_dir = getattr(args, 'out_dir', None)
        self.inputs, self.outputs = [], []
        self.started = datetime.now()

        run = self.cfg['run']
        self.name = run['name']
        # Relative paths are relative to the config file, so a run does not depend on the cwd
        config_dir = os.path.dirname(self.config_path)
        self.data_root = os.path.normpath(os.path.join(config_dir, os.path.expanduser(run['data_root'])))
        self.run_dir = os.path.normpath(os.path.join(config_dir, os.path.expanduser(
            run.get('run_dir', '{data_root}/{name}').format(data_root=self.data_root, name=self.name))))
        self.ref_time = run.get('ref_time', '2014-01-01')
        self._freeze(getattr(args, 'update_config', False))

    # %% Settings

    def section(self, name):
        """A config table, e.g. run.section('L1') or run.section('P1.bgc'); {} if missing."""
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

    # %% Files

    def _template(self, key, dataset=None):
        files = self.cfg.get('files', {})
        for k in ([f'{key}_{dataset}'] if dataset else []) + [key]:
            if k in files:
                return files[k]
        for k in ([f'{key}_{dataset}'] if dataset else []) + [key]:
            if k in DEFAULT_FILES:
                return DEFAULT_FILES[k]
        raise SystemExit(f'{self.config_path}: no [files] entry for "{key}"'
                         + (f' (or "{key}_{dataset}")' if dataset else ''))

    def path(self, key, dataset=None, **fmt):
        """Resolved path (or list of paths) for a [files] key, with no existence checks."""
        template = self._template(key, dataset)
        def resolve(t):
            p = os.path.expanduser(t.format(data_root=self.data_root, run_dir=self.run_dir,
                                            dataset=dataset, **fmt))
            return os.path.normpath(os.path.join(os.path.dirname(self.config_path), p))
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

    def feature_files(self, years):
        """Gridded P1 feature files for p1_features.add_all_features(), as paths with {yr};
        checks every year's file exists first."""
        keys = {'slp': 'era5_slp', 'wind': 'era5_wind', 'adt': 'adt', 'seaice': 'seaice'}
        for key in keys.values():
            for yr in years:
                self.input(key, yr=yr)
        return {name: self.template(key) for name, key in keys.items()} | {'mbl': self.input('mbl')}

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

    # %% Provenance

    def _freeze(self, update):
        """Copy the config into the run folder, or check it matches the copy already there."""
        frozen = os.path.join(self.run_dir, 'config.toml')
        if os.path.exists(frozen):
            if os.path.samefile(frozen, self.config_path):
                return
            with open(frozen, 'rb') as f:
                same = tomllib.load(f) == self.cfg
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
