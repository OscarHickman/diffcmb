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
# D5 (2026-09-25): the certified Block-4-ON ensemble is nu = 30, 1000 + 3600
# sweeps (ROADMAP T0.1c/d); its blind_rNNN.npz pair on the same skies.
ENSEMBLE=results/analysis/ens_exact_l64_A3000_n30_nu30_long
# Figure 1 thin ~ a_lm tau_int (median Block 2 ESS 77 per 3600 sweeps -> tau ~ 47),
# so the rank draws are close to independent. The a_lm null must be made at the
# same thin: scripts/submit_null_alm_power_rank.slurm with THIN=${FIG1_THIN}.
FIG1_THIN=50
figures:
	${FIG_ENV} ${PYTHON} scripts/paper/fig_schematic.py
	${FIG_ENV} ${PYTHON} scripts/paper/fig_maps.py --chain ${ENSEMBLE}/chain_r000.npz
	${FIG_ENV} ${PYTHON} scripts/paper/fig_convergence.py --indir ${ENSEMBLE}
	${FIG_ENV} ${PYTHON} scripts/paper/fig1_validation.py --indir ${ENSEMBLE} --thin ${FIG1_THIN}
	${FIG_ENV} ${PYTHON} scripts/paper/fig4_joint_posterior.py --indir ${ENSEMBLE} --thin 45
	${FIG_ENV} ${PYTHON} scripts/paper/fig2_bias_reduction.py --indir ${ENSEMBLE}
	${FIG_ENV} ${PYTHON} scripts/paper/fig3_uncertainty_propagation.py --indir ${ENSEMBLE}

clean:
	rm -rf ${venv}