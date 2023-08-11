#!/bin/bash
#SBATCH --account=def-simine
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-5
#SBATCH --time=0-10:00
#SBATCH --output=slurm-%a.out
#SBATCH --error=slurm-%a.err

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

n=$SLURM_ARRAY_TASK_ID

# File to read
file_path="../explicit_percolation/successful_npys.txt"

# Read lines from the file into an array
lines=()
while IFS= read -r line; do
    lines+=("$line")
done < "$file_path"

nlines=174
njobs=6

nlines_per_job=$((nlines/njobs))


i0=$((nlines_per_job * n))
ilast=$((nlines_per_job * (n+1) -1))

line_inds=$(seq $i0 $ilast)

for i in ${line_inds[@]} ; do

    j=${lines[i]}
    echo "~ ~ ~ ~ ~ ~ ~ ~ ~ ~ Working on sample $j ~  ~  ~  ~  ~  ~  ~  ~  ~  ~ "
    workdir="sample-$j"
    
    if [ ! -d $workdir ]; then

        mkdir $workdir
    fi 

    cp scripts/* $workdir
    cd $workdir
    
    srun python run_MCpercolate_narval.py $j 

    cd ..
done
