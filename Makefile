venv=.venv
PYTHON=${venv}/bin/python
PIP=${venv}/bin/pip
ACTIVATE=. ${venv}/bin/activate

.PHONY: all setup install precommit examples test build-rust clean

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
	cd src/rust_sph && maturin develop --release

precommit:
	${PYTHON} -m pre_commit run --all-files

examples:

test:
	# run tests inside venv with src on PYTHONPATH
	PYTHONPATH=src ${PYTHON} -m pytest -q

clean:
	rm -rf ${venv}