# Submitting contributions

This repository is setup to only accept contributions as pull
requests.  So if you have a code suggestion, fork the repo (or create
a branch), and open a PR.  The PR should be reviewed before it can be
merged.

# Development setup

We separate out the modelling backends into optional groups
(e.g. `pypsa`).  So depending on the model someone wants to work with,
you have to choose the appropriate backend.

We use `hatch` to manage environments, and build a python package.
However it is possible to also manage the environments yourself and
build/install with `pip`.

_Note:_ some pytest options are enabled by default, see the
`tool.pytest.ini_options` section in `pyproject.toml`.

## Using w/ `hatch`

- Show available environments:
  ```
  $ hatch env show
  ```
- Create an environment:
  ```
  $ hatch env create <name>
  ```
- Run a command in an environment:
  ```
  $ hatch run <name>:<command>
  ```
  So you can run tests or linting as:
  ```
  $ hatch run test:pytest [options]
  $ hatch run lint:black
  $ hatch run types:check-mypy
  ```
- Build a python package:
  ```
  $ hatch build
  ```
  The source tarball and wheel file is created in `dist/`.

## Managing environments manually

- Create a virtual environment:
  ```
  $ python -m venv --prompt MGA4all venv
  ```
- Activate the virtual environment:
  ```
  $ source venv/bin/activate   # linux/macos
  $ ./bin/Scripts/activate.bat # windows default shell
  $ ./bin/Scripts/activate.ps1 # powershell
  ```
- Install in editable mode:
  ```
  $ pip install -e .[pypsa,dev]
  ```
- Run tests:
  ```
  $ pytest [options]
  ```
