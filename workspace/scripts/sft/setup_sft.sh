set -x
set -e

# Determine repo root (two levels up from scripts/sft/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# Uninstall container-shipped verl and install from local directory
pip uninstall -y verl || true

cd "${REPO_ROOT}/verl"
pip install -e .
cd -

# Verify local verl is now active
python -c "import verl; print(f'verl version: {verl.__version__}')"
pip freeze | grep verl
pip freeze | grep torch
pip freeze | grep transformers

pip install tensordict
