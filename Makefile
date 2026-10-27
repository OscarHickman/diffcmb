venv=.venv
PYTHON=${venv}/bin/python
PIP=${venv}/bin/pip
ACTIVATE=. ${venv}/bin/activate

.PHONY: all setup install precommit test build-rust clean figures figure-npe figure-realdata figure-lcdm

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
# same thin: scripts/slurm/submit_null_alm_power_rank.slurm with THIN=${FIG1_THIN}.
FIG1_THIN=50
# D7 (2026-10-14): figure 1 is the strict SBC of all four blocks on the strict
# ensemble (proper nu = 30 priors on C_l^TT and C_L^phiphi, truths drawn from
# them, same phi truths as ENSEMBLE), read against uniform with no simulated
# null. The flat-prior ENSEMBLE's figure 1 (a_lm against its exact-sampler null)
# is kept as figure1_flatprior for the caption.
STRICT_ENSEMBLE=results/analysis/ens_exact_l64_A3000_n30_nu30_strict
PLOTS=/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots
# 2026-10-26: the paper repo was renumbered into plots/figureN/ (order of appearance) and a flat
# plots/appendix/, as papers/1_2PCF: schematic -> figure1, maps -> figure2, figure1 -> figure3,
# figure2 -> figure4, figure3 -> figure5, figure4 -> figure6, npe -> figure7 + appendix,
# realdata -> figure8, lcdm -> figure9, convergence -> appendix. The flat-prior figure 1 is not
# in the paper (caption numbers only), so it is written here, outside the paper repo.
figures:
	${FIG_ENV} ${PYTHON} scripts/paper/fig_schematic.py
	${FIG_ENV} ${PYTHON} scripts/paper/fig_maps.py --chain ${ENSEMBLE}/chain_r000.npz
	${FIG_ENV} ${PYTHON} scripts/paper/fig_convergence.py --indir ${ENSEMBLE} --rank_thin ${FIG1_THIN}
	${FIG_ENV} ${PYTHON} scripts/paper/fig1_validation.py --indir ${STRICT_ENSEMBLE} --thin ${FIG1_THIN} --outdir ${PLOTS}/figure3
	${FIG_ENV} ${PYTHON} scripts/paper/fig1_validation.py --indir ${ENSEMBLE} --thin ${FIG1_THIN} --outdir plots/paper_unused/figure1_flatprior
	${FIG_ENV} ${PYTHON} scripts/paper/fig4_joint_posterior.py --indir ${ENSEMBLE} --thin 45
	${FIG_ENV} ${PYTHON} scripts/paper/fig2_bias_reduction.py --indir ${ENSEMBLE}
	${FIG_ENV} ${PYTHON} scripts/paper/fig3_uncertainty_propagation.py --indir ${ENSEMBLE}

# ROADMAP T1.1 (D6, D9): the baseline flow against the strict exact ensemble.
NPE_FLOW=results/analysis/npe_sims_l64_A3000_n30_nu30/flow_maf5x256
figure-npe:
	${FIG_ENV} ${PYTHON} scripts/paper/fig_npe.py --flow ${NPE_FLOW} --outdir ${PLOTS}/appendix
	# The three headline panels are main-text figure 7; the other four stay in the appendix.
	mkdir -p ${PLOTS}/figure7
	for f in npe_width_ratio npe_width_ratio_ell npe_shift; do if [ -f ${PLOTS}/appendix/$$f.pdf ]; then mv ${PLOTS}/appendix/$$f.pdf ${PLOTS}/figure7/; fi; done

# ROADMAP T1.2a: needs the realdata_summary.py outputs of the three real-data runs.
REALDATA=results/analysis/realdata
# The C_L^phiphi panel refuses the A_phi = 1 run until the phi trap is fixed (ROADMAP T1.2a);
# `make figure-realdata PANELS=cltt` builds the C_l^TT panel alone. Both A_phi = 1 maps are
# pooled over the same 8 noise draws (realdata_summary.py --chains <run> <run>_nr1 ... _nr7),
# since the draw-to-draw scatter is as large as SMICA vs SEVEM below l = 60.
PANELS=cltt,clpp
figure-realdata:
	${FIG_ENV} ${PYTHON} scripts/paper/fig_realdata.py --panels ${PANELS} --a1 ${REALDATA}_smica_l64_A1_n30_clnu30_pooled/summary.npz --a3000 ${REALDATA}_smica_l64_A3000_n30_clnu30/summary.npz --sevem ${REALDATA}_sevemR4_l64_A1_n30_clnu30_pooled/summary.npz --outdir ${PLOTS}/figure8

# ROADMAP T1.4: needs scripts/slurm/submit_br_lcdm.slurm's outputs (calibrations with oracle rows,
# and SMICA / PR4 SEVEM at A_phi 1 over 8 noise draws).
figure-lcdm:
	${FIG_ENV} ${PYTHON} scripts/paper/fig_lcdm.py --indir results/analysis/br_lcdm --outdir ${PLOTS}/figure9

clean:
	rm -rf ${venv}