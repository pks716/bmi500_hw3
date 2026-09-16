
import time


def _tic():
    """Start a timer for a code section."""
    return time.time()


def _toc(label, start_time):
    """Print the elapsed wall-clock time for a code section."""
    elapsed = time.time() - start_time
    print(f"[TIMING] {label}: {elapsed:.3f}s")


_t0 = _tic()
_toc("timing setup", _t0)

# %%
_t0 = _tic()

from typing_extensions import ParamSpecArgs
import numpy as np
import pandas as pd
import scanpy as sc

import sys
import argparse

_toc("imports", _t0)

# %%
_t0 = _tic()

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
# sc.settings.n_jobs = int(sys.argv[4])
sc.settings.n_jobs = 1

print(f"using {sc.settings.n_jobs} threads")

_toc("scanpy settings", _t0)

# %%
_t0 = _tic()

parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('--data-dir', type=str, help='Directory containing the dataset subdirectories', default='data')
parser.add_argument('--data-set', type=str, help='Dataset name, which is the subdirectory name', default='pbmc3k')
parser.add_argument('--out-dir', type=str, help='Output directory', required=False, default='data')
parser.add_argument('--num-threads', type=int, help='Number of threads', default=1, required=False)

args = parser.parse_args()

datadir = args.data_dir if args.data_dir.endswith('/') else args.data_dir + '/'
dataset = args.data_set
outdir = args.out_dir if args.out_dir.endswith('/') else args.out_dir + '/'
nthreads = args.num_threads

_toc("argument parsing", _t0)

# %%
_t0 = _tic()

# I/O
results_file = "/".join([outdir, dataset + '.scanpy.h5ad'])  # the file that will store the analysis results

adata = sc.read_10x_mtx(
    #'/nethome/tpan7/scgc/data/' + dataset + '/filtered_gene_bc_matrices/hg19',  # the directory with the `.mtx` file
    "/".join([datadir, dataset, 'filtered_gene_bc_matrices']),  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)                              # write a cache file for faster subsequent reading

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`

_toc("I/O: read 10x data", _t0)

# %%
_t0 = _tic()

# preprocessing
# basic filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

_toc("preprocessing: basic filtering", _t0)

# %%
_t0 = _tic()

# metric
#adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
#sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# filtering by slicing the AnnData object
#adata = adata[adata.obs.n_genes_by_counts < 2500, :]
#adata = adata[adata.obs.pct_counts_mt < 5, :]


# and normalize to 10K reads per cell
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

_toc("QC metrics + normalization", _t0)

# %%
_t0 = _tic()

# highly variable genes

#sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=2000)

# freeze data.
adata.raw = adata

# filtering by highly variable genes.
adata = adata[:, adata.var.highly_variable]

_toc("highly variable genes", _t0)

# %%
_t0 = _tic()

# regres out effects of total counts per cell an d% mitochondrial genes
#sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)

_toc("scale", _t0)

# %%
_t0 = _tic()

# report adata - so we can check ot see if we are comparable to Seurat
# adata.write(results_file)
# adata

_toc("report adata (no-op)", _t0)

# %%
_t0 = _tic()

# pca.  parallel via OMP_NUM_THREADS
sc.tl.pca(adata, svd_solver='arpack', n_comps=30)

# adata.write(results_file)
# adata

_toc("PCA", _t0)

# %%
_t0 = _tic()

# neighborhood graph
sc.pp.neighbors(adata, n_pcs=30)

_toc("neighbors", _t0)

# %%
_t0 = _tic()

# for fixing disconnected clusters or connectivity issues:
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#cs.tl.umap(adata, init_pos='paga')


# adata.write(results_file)
# adata

_toc("paga fix (no-op)", _t0)

# %%
_t0 = _tic()

# clustering  (currently uses leiden,  previously using louvain (like Seurat).)
#sc.tl.leiden(adata)
sc.tl.louvain(adata, resolution=0.5)

_toc("clustering: louvain", _t0)

# %%
_t0 = _tic()

# umap
sc.tl.umap(adata, n_components=30)

_toc("umap", _t0)

# %%
_t0 = _tic()

adata.write(results_file)
adata

_toc("write results file", _t0)

# %%
_t0 = _tic()

# support t-test, wilcoxon, logistic regression
# find marker genes
import cProfile
import pstats

_profile_output = "rank_genes_groups.prof"
cProfile.run(
    "sc.tl.rank_genes_groups(adata, 'louvain', method='wilcoxon', use_raw=True)",
    _profile_output,
)

# Print the top functions by cumulative time straight to the console
stats = pstats.Stats(_profile_output)
stats.sort_stats("cumulative").print_stats(15)

_toc("rank_genes_groups (wilcoxon, cProfile'd)", _t0)
