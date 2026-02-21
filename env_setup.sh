#!/bin/bash
set -euo pipefail

env_name=graphmert


# Make conda activation work in scripts:
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # typical Miniconda
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # typical Anaconda
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  # fallback: ask conda where its base is
  CONDA_BASE="$(conda info --base)"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi

conda create -y -n "$env_name" python=3.13
conda activate "$env_name" || { echo "Failed to activate env: $env_name" >&2; exit 1; }

export PYTHONNOUSERSITE=1
unset PIP_USER
python -m pip install -U pip

python -m pip install --no-input \
      torch transformers cython apache-beam datasets evaluate spacy accelerate
python -m spacy download en_core_web_sm
