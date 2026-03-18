import tpunanny as tn

project_id = 'loss-spikes'

tn.babysit(
    idxs=[0], # using a single TPU, index 0
    tpu_type='v6e-8',
    zone='us-east1-d',
    project_id=project_id,
    ssh_script='echo "hello world"',
    startup_script='#!/bin/bash\necho "startup script ran" >> /tmp/startup.log',
    ensure_nat=True
)