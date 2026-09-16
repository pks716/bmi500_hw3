#!/bin/bash
#SBATCH --job-name=scanpy_pbmc6k
#SBATCH --output=logs/pbmc6k_%j.out
#SBATCH --error=logs/pbmc6k_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=<partition>      # use available partitions with: sinfo
#SBATCH --account=<account>          # redacted for privacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

#environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bmi500

# /usr/bin/time -v gives coarse-grain stats (wall time, CPU time, peak memory)
# the script itself prints [TIMING] lines per #%% section (instrumented)
# and runs cProfile on rank_genes_groups (fine-grain), saved to rank_genes_groups.prof
/usr/bin/time -v python scanpy_pbmc_time.py \
    --data-dir data \
    --data-set pbmc6k \
    --out-dir out

# keep this dataset's fine-grain profile separate from the others
mv rank_genes_groups.prof pbmc6k_rank_genes_groups.prof
