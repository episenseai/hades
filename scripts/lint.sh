#!/bin/sh

set -e
set -x

mypy carbon
black carbon tests -S -l 120 --check --diff
isort carbon tests scripts --check-only
flake8 carbon tests
pylint carbon tests
