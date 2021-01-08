#!/usr/bin/env bash

set -e
set -x

mypy carbon
flake8 carbon tests
pylint carbon tests
yapf carbon tests --diff --recursive
isort carbon tests scripts --check-only
