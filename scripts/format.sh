#!/usr/bin/env bash

set -e
set -x

yapf carbon tests --in-place --recursive
isort carbon tests scripts
