import os
import math
import tqdm
import traceback
import s3fs
import uproot
import awkward as ak
from .tools import _concat
from ..logger import _logger

def get_s3_client():
    s3_endpoint = os.getenv('S3_ENDPOINT', 'https://s3.cern.ch')
    s3 = s3fs.core.S3FileSystem(client_kwargs={'endpoint_url': s3_endpoint})
    return s3

def _read_hdf5(filepath, branches, load_range=None):
    import tables
    tables.set_blosc_max_threads(4)
    with tables.open_file(filepath) as f:
        outputs = {k:getattr(f.root, k)[:] for k in branches}
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs[branches[0]]))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs


def _read_root(filepath, branches, load_range=None, treename=None):
    with uproot.open(filepath) as f:
        if treename is None:
            treenames = set([k.split(';')[0] for k, v in f.items() if getattr(v, 'classname', '') == 'TTree'])
            if len(treenames) == 1:
                treename = treenames.pop()
            else:
                raise RuntimeError('Need to specify `treename` as more than one trees are found in file %s: %s' % (filepath, str(branches)))
        tree = f[treename]
        if load_range is not None:
            start = math.trunc(load_range[0] * tree.num_entries)
            stop = max(start + 1, math.trunc(load_range[1] * tree.num_entries))
        else:
            start, stop = None, None
        outputs = tree.arrays(branches, entry_start=start, entry_stop=stop)
    return outputs


def _read_awkd(filepath, branches, load_range=None):
    with ak.load(filepath) as f:
        outputs = {k: f[k] for k in branches}
    if load_range is not None:
        start = math.trunc(load_range[0] * len(outputs[branches[0]]))
        stop = max(start + 1, math.trunc(load_range[1] * len(outputs[branches[0]])))
        for k, v in outputs.items():
            outputs[k] = v[start:stop]
    return outputs


def _read_files(filelist, branches, load_range=None, show_progressbar=False, **kwargs):
    import os
    from collections import defaultdict
    branches = list(branches)
    table = defaultdict(list)
    if show_progressbar:
        filelist = tqdm.tqdm(filelist)
    s3 = None
    if filelist[0].startswith('s3'):
        s3 = get_s3_client()
        filelist = sum([s3.glob(f) for f in filelist], [])
    for filepath in filelist:
        ext = os.path.splitext(filepath)[1]
        if ext not in ('.h5', '.root', '.awkd'):
            raise RuntimeError('File %s of type `%s` is not supported!' % (filepath, ext))
        if s3:
            filepath = s3.open(filepath)
        try:
            if ext == '.h5':
                a = _read_hdf5(filepath, branches, load_range=load_range)
            elif ext == '.root':
                a = _read_root(filepath, branches, load_range=load_range, treename=kwargs.get('treename', None))
            elif ext == '.awkd':
                a = _read_awkd(filepath, branches, load_range=load_range)
        except Exception as e:
            a = None
            _logger.error('When reading file %s:', filepath)
            _logger.error(traceback.format_exc())
        if s3:
            filepath.close()
        if a is not None:
            for name in branches:
                table[name].append(ak.values_astype(a[name], 'float32'))
    table = {name:_concat(arrs) for name, arrs in table.items()}
    if len(table[branches[0]]) == 0:
        raise RuntimeError(f'Zero entries loaded when reading files {filelist} with `load_range`={load_range}.')
    return table


def _write_root(file, table, treename='Events', compression=-1, step=1048576):
    if compression == -1:
        compression = uproot.LZ4(4)
    with uproot.recreate(file, compression=compression) as fout:
        fout.mktree(treename, {k:v.dtype for k, v in table.items()})
        # fout[treename] = uproot.newtree({k:v.dtype for k, v in table.items()})
        start = 0
        while start < len(list(table.values())[0]) - 1:
            fout[treename].extend({k:v[start:start + step] for k, v in table.items()})
            start += step
