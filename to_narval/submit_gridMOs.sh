#!/bin/bash
#SBATCH --account=ctb-simine
#SBATCH --time=0-12:00
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-7
#SBATCH --output=slurm_%a.out
#SBATCH --error=slurm_%a.err


virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt

nn=$SLURM_ARRAY_TASK_ID

if [[ ! -d sample-${nn} ]]; then
	mkdir sample-${nn}
	#cp 20x20_gridMOs/sample-${nn}/gam*npy sample-${nn}
else
	rm "sample-${nn}/*.py"
	rm "sample-${nn}/*.pkl"
	sleep 20
fi

cp run_gMOs.py sample-${nn}
cd sample-${nn}

python run_gMOs.py $nn 'tdot25' 'virtual'

