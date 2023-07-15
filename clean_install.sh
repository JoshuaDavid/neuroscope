#!/bin/bash

set -euxo pipefail

# Wipe out anything that was here before
rm -rf venv

python -m venv venv
source venv/bin/activate
pip install .
