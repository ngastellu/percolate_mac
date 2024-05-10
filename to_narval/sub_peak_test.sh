#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --time=0-01:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-1
#SBATCH --output=slurm_%a.out
#SBATCH --error=slurm_%a.err


virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt


run_inds=(6 34)

n=$SLURM_ARRAY_TASK_ID

nn=${run_inds[n]}


if [[ ! -d sample-${nn} ]]; then
	mkdir sample-${nn}
else
	rm "sample-${nn}/*.py"
	rm "sample-${nn}/*.pkl"
	sleep 20
fi

cp test_sites.py sample-${nn}
cd sample-${nn}

python test_sites.py $nn 

