#!/bin/sh

set -e
set -x

mypy carbon
black carbon tests -S -l 120 --check --diff
isort carbon tests scripts --check-only
flake8 carbon tests
pylint carbon tests
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --check carbon tests scripts --exclude=__init__.py
