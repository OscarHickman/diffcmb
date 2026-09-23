venv=.venv
PYTHON=${venv}/bin/python
PIP=${venv}/bin/pip
ACTIVATE=. ${venv}/bin/activate

.PHONY: all setup install precommit examples test build-rust clean figures

all: setup

setup: ${venv}/bin/activate
	${PIP} install --upgrade pip
	${PIP} install -r requirements.txt
	# ensure pre-commit hooks are available
	${PIP} install pre-commit ruff || true
	${PYTHON} -m pre_commit install || true

${venv}/bin/activate:
	python -m venv ${venv}
	@echo "created venv at ${venv}"

build-rust:
	cd diffcmb/rust_sph && maturin develop --release

precommit:
	${PYTHON} -m pre_commit run --all-files

examples:

test:
	# run tests inside venv with src on PYTHONPATH
	PYTHONPATH=diffcmb ${PYTHON} -m pytest -q

# Paper figures (papers/7_DiffCMB/plots/). Journal preset: DIFFCMB_JOURNAL=PRD|MNRAS|JCAP.
FIG_ENV=PYTHONPATH=diffcmb:scripts:scripts/paper OMP_NUM_THREADS=4
ENSEMBLE=results/analysis/ens_exact_l64_A3000_n30
figures:
	${FIG_ENV} ${PYTHON} scripts/paper/fig_schematic.py
	${FIG_ENV} ${PYTHON} scripts/paper/fig_maps.py
	${FIG_ENV} ${PYTHON} scripts/paper/fig_convergence.py
	${FIG_ENV} ${PYTHON} scripts/paper/fig1_validation.py --thin 10
	${FIG_ENV} ${PYTHON} scripts/paper/fig4_joint_posterior.py --indir ${ENSEMBLE} --thin 45
	${FIG_ENV} ${PYTHON} scripts/paper/fig2_bias_reduction.py --indir ${ENSEMBLE}
	${FIG_ENV} ${PYTHON} scripts/paper/fig3_uncertainty_propagation.py --indir ${ENSEMBLE}

clean:
	rm -rf ${venv}