from pathlib import Path

import tpunanny as tn

project_id = 'loss-spikes'
base_dir = Path(__name__).resolve().parent

tn.babysit(
    idxs=[0], # using a single TPU, index 0
    tpu_type='v6e-16',
    zone='us-east1-d',
    project_id=project_id,
    ssh_script=(base_dir / 'run.sh').read_text(),
    # ssh_script='cd loss-spikes-project/picodo && git pull',
    # ssh_script="pkill -9 python3 || true",
    startup_script=(base_dir / 'multi_vm_tpu_setup.sh').read_text(),
    ensure_nat=True
)
