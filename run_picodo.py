import os
import shlex
from pathlib import Path

import tpunanny as tn


def _read_env_key(path, key):
    prefix = f'{key}='
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line[len(prefix):].strip().strip("'\"")
    return None


project_id = 'loss-spikes'
base_dir = Path(__file__).resolve().parent
wandb_token = os.environ.get('WANDB_TOKEN') or _read_env_key(base_dir / '.env', 'WANDB_TOKEN')
if not wandb_token:
    raise RuntimeError('WANDB_TOKEN is missing. Set it in .env or export it in your shell.')

startup_script = (base_dir / 'multi_vm_tpu_setup.sh').read_text()
startup_script = f"export WANDB_TOKEN={shlex.quote(wandb_token)}\n{startup_script}"

tn.babysit(
    idxs=[0], # using a single TPU, index 0
    tpu_type='v6e-16',
    zone='us-east1-d',
    project_id=project_id,
    ssh_script=(base_dir / 'run.sh').read_text(),
    # ssh_script='cd loss-spikes-project/picodo && git pull',
    # ssh_script="pkill -9 python3 || true",
    startup_script=startup_script,
    ensure_nat=True
)
