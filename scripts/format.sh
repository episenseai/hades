#!/bin/sh

set -e
set -x

# autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place carbon tests scripts --exclude=__init__.py
black carbon tests -l 120
isort carbon tests scripts
