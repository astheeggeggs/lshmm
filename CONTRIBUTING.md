# Contributing

## Setting up a development environment

- [Install pre-commit](https://pre-commit.com/#installation) and setup pre-commit hooks.

  `pre-commit install`

- Install [Nox](https://nox.thea.codes/).

  `pip install nox`

## Conventions

Code should be formatted with [Black](https://black.readthedocs.io/).
Imports should be sorted with [isort](https://pycqa.github.io/isort/).

To format code and sort imports, use `nox -rs format`.

Docstrings should conform to [PEP 257](https://www.python.org/dev/peps/pep-0257/).

To check for conformance using [pydocstyle](https://www.pydocstyle.org/), use `nox -rs pydocstyle`.

## Pylint

[Pylint](https://www.pylint.org/) can catch some types of errors without running a pipeline.

To run Pylint, use `nox -rs lint`

## Tests

To run tests, use `nox -rs test`

Use [pytest](https://pytest.org/) to run tests.

## Documentation

To build documentation with [Sphinx](https://www.sphinx-doc.org/), use `nox -rs build_docs`.

reStructuredText references:

- [Sphinx reStructuredText documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html)
- [docutils reStructuredText documentation](https://docutils.sourceforge.io/rst.html)

To view documentation (after building it), use `nox -rs serve_docs` and open http://localhost:8000 in a web browser.

## Dependencies

List dependencies in the [`install_requires`](https://packaging.python.org/discussions/install-requires-vs-requirements/) section of `setup.cfg`.

For development dependencies, list direct dependencies in `requirements.dev.in` and use [pip-compile](https://github.com/jazzband/pip-tools) to generate `requirements.dev.txt` (`nox -rs pip_compile -- requirements.dev.in`).
