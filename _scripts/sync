#!/bin/bash
find . -name '.ipynb_checkpoints' -prune -o -name '*.ipynb' -print | xargs jupytext --from ipynb --to py:percent
