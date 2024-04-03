## Installation

We set up a devcontainer, so that when people open the repo in GitHub codespaces everything is set up. Can run in local devcontainer as well.

Otherwise, you should run this command
```
pip install -e .[dev]
```

## Running tests

The previous install should install optional dependencies `pytest` and `pytest-cov`

```sh
# test
pytest
# with coverage reporting
pytest --cov=src tests/
# get the xml
pytest --cov=src tests/ --cov-report xml
```

In the devcontainer we have act installed, allowing us to verify that pytest runs in the CI/CD pipeline as well.
```
act -j build -s "CODECOV_TOKEN=your-codecov-token-abc555-5555"
```