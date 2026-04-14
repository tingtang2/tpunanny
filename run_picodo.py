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


def _parse_seed_zone_map(raw):
    mapping = {}
    if not raw:
        return mapping

    entries = [entry.strip() for entry in raw.split(',') if entry.strip()]
    for entry in entries:
        if '=' not in entry:
            raise RuntimeError(
                f'Invalid SEED_ZONE_MAP entry "{entry}". Expected format like "0=us-east1-d".'
            )
        seed_text, zone = entry.split('=', 1)
        mapping[int(seed_text.strip())] = zone.strip()
    return mapping


def _sanitize_exp_tag(raw):
    cleaned = ''.join(ch.lower() if ch.isalnum() else '-' for ch in raw.strip())
    while '--' in cleaned:
        cleaned = cleaned.replace('--', '-')
    cleaned = cleaned.strip('-')
    return cleaned or 'default'


project_id =  os.environ.get('GCP_PROJECT_ID', 'loss-spikes')
tpu_type =  os.environ.get('TPU_TYPE', 'v6e-16')
base_dir = Path(__file__).resolve().parent
wandb_token = os.environ.get('WANDB_TOKEN') or _read_env_key(base_dir / '.env', 'WANDB_TOKEN')
if not wandb_token:
    raise RuntimeError('WANDB_TOKEN is missing. Set it in .env or export it in your shell.')
hf_token = os.environ.get('HF_TOKEN') or _read_env_key(base_dir / '.env', 'HF_TOKEN')
run_name_prefix = os.environ.get('RUN_NAME_PREFIX', 'exp_gpt3xl')
exp_tag = _sanitize_exp_tag(os.environ.get('EXP_TAG', run_name_prefix))
tpu_id_prefix = f'tn-{exp_tag}'
if len(tpu_id_prefix) > 24:
    # Keep room for "-{tpu_type}-{seed}" under TPU name length limits.
    tpu_id_prefix = tpu_id_prefix[:24].rstrip('-')
run_script = (base_dir / 'run.sh').read_text()
remote_runner_script_path = '/home/tingchen/loss-spikes-project/tpunanny_run.sh'


def _build_remote_runner_script(script_text):
    remote_runner_dir = os.path.dirname(remote_runner_script_path)
    remote_runner_tmp_path = f'{remote_runner_script_path}.tmp'
    return (
        f"mkdir -p {shlex.quote(remote_runner_dir)}\n"
        f"cat > {shlex.quote(remote_runner_tmp_path)} <<'__TPUNANNY_RUN_SH__'\n"
        f"{script_text}\n"
        "__TPUNANNY_RUN_SH__\n"
        f"mv {shlex.quote(remote_runner_tmp_path)} {shlex.quote(remote_runner_script_path)}\n"
        f"chmod +x {shlex.quote(remote_runner_script_path)}\n"
        f"export TPUNANNY_RUNNER_SCRIPT_PATH={shlex.quote(remote_runner_script_path)}"
    )


remote_runner_bootstrap = _build_remote_runner_script(run_script)

num_seeds = int(os.environ.get('NUM_SEEDS', '5'))
seed_start = int(os.environ.get('SEED_START', '0'))
seed_idxs = list(range(seed_start, seed_start + num_seeds))

default_zone = os.environ.get('DEFAULT_ZONE', 'us-east1-d')
zones_csv = os.environ.get('SEED_ZONES', default_zone)
zones = [zone.strip() for zone in zones_csv.split(',') if zone.strip()]
if not zones:
    raise RuntimeError('SEED_ZONES resolved to an empty list.')
seed_zone_map = _parse_seed_zone_map(os.environ.get('SEED_ZONE_MAP', ''))

# train config setup
lr_tag = os.environ.get('LR_TAG', 'default')
lr_arg = os.environ.get('LR_ARG', '')
checkpoint_root = os.environ.get('CHECKPOINT_ROOT')
batch_size = os.environ.get('BATCH_SIZE', '64')
num_tp_devices = os.environ.get('NUM_TP_DEVICES', '1')
wandb_mode = os.environ.get('WANDB_MODE', 'online')
config_name = os.environ.get('CONFIG_NAME', 'wortsman_default')
checkpoint_freq = os.environ.get('CHECKPOINT_FREQ', '10000')
eval_freq = os.environ.get('EVAL_FREQ', '100')
use_chinchilla = os.environ.get('USE_CHINCHILLA', 'false')
use_z_loss = os.environ.get('USE_Z_LOSS', '')
use_b2_cosine_anneal = os.environ.get('USE_B2_COSINE_ANNEAL', '')
b2_arg = os.environ.get('B2_ARG', '')
final_b2_arg = os.environ.get('FINAL_B2_ARG', '')
follow_logs = os.environ.get('FOLLOW_LOGS', 'false').strip().lower() == 'true'
follow_log_lines = os.environ.get('FOLLOW_LOG_LINES', '200')
sequential_seeds_on_single_tpu = (
    os.environ.get('SEQUENTIAL_SEEDS_ON_SINGLE_TPU', 'false').strip().lower() == 'true'
)

seed_zone_by_idx = {}
for offset, seed in enumerate(seed_idxs):
    seed_zone_by_idx[seed] = seed_zone_map.get(seed, zones[offset % len(zones)])

zones_by_idx = {}
ssh_script_by_idx = {}
follow_logs_command_by_idx = {}
healthcheck_command_by_idx = {}
completion_command_by_idx = {}
checkpoint_root_by_region = {}
babysit_idxs = seed_idxs

if sequential_seeds_on_single_tpu:
    controller_seed = seed_idxs[0]
    controller_zone = seed_zone_by_idx[controller_seed]
    zones_by_idx[controller_seed] = controller_zone
    controller_region = controller_zone.rsplit('-', 1)[0]

    unique_zones = sorted(set(seed_zone_by_idx.values()))
    if len(unique_zones) > 1:
        print(
            '[run_picodo] SEQUENTIAL_SEEDS_ON_SINGLE_TPU=true ignores per-seed zone overrides; '
            f'using only {controller_zone} for all seeds {seed_idxs}. Requested zones: {unique_zones}'
        )

    if checkpoint_root:
        shared_checkpoint_root = checkpoint_root
    else:
        fineweb_bucket = tn.get_fineweb_bucket_name(controller_zone, project_id)
        shared_checkpoint_root = f'gs://{fineweb_bucket}/picodo_ckpts'

    seed_queue_csv = ','.join(str(seed) for seed in seed_idxs)
    seed_queue_slug = '-'.join(str(seed) for seed in seed_idxs)
    queue_session_name = f'picodo_train_queue_seed{seed_idxs[0]}_to{seed_idxs[-1]}'
    queue_run_name = f'{run_name_prefix}_seeds{seed_queue_slug}_lr{lr_tag}_bs{batch_size}'
    queue_done_marker = f'/home/tingchen/loss-spikes-project/picodo/train_status/{queue_run_name}.done'
    queue_log_path = f'/home/tingchen/loss-spikes-project/picodo/train_logs/{queue_run_name}.log'

    env_exports = {
        'SEED': str(controller_seed),
        'LR_TAG': lr_tag,
        'RUN_NAME_PREFIX': run_name_prefix,
        'CHECKPOINT_ROOT': shared_checkpoint_root,
        'BATCH_SIZE': batch_size,
        'NUM_TP_DEVICES': num_tp_devices,
        'WANDB_MODE': wandb_mode,
        'CONFIG_NAME': config_name,
        'CHECKPOINT_FREQ': checkpoint_freq,
        'EVAL_FREQ': eval_freq,
        'USE_CHINCHILLA': use_chinchilla,
        'USE_Z_LOSS': use_z_loss,
        'USE_B2_COSINE_ANNEAL': use_b2_cosine_anneal,
        'B2_ARG': b2_arg,
        'FINAL_B2_ARG': final_b2_arg,
        'WANDB_TOKEN': wandb_token,
        'SEQUENTIAL_SEEDS_ON_SINGLE_TPU': 'true',
        'SEED_QUEUE': seed_queue_csv,
        'SEED_QUEUE_SESSION_NAME': queue_session_name,
        'SEED_QUEUE_DONE_MARKER': queue_done_marker,
        'SEED_QUEUE_LOG_FILE': queue_log_path,
    }
    if hf_token:
        env_exports['HF_TOKEN'] = hf_token
    if lr_arg:
        env_exports['LR_ARG'] = lr_arg

    exports_text = '\n'.join(
        f"export {key}={shlex.quote(value)}"
        for key, value in env_exports.items()
    )
    ssh_script_by_idx[controller_seed] = (
        f"{exports_text}\n"
        f"{remote_runner_bootstrap}\n"
        f"bash {shlex.quote(remote_runner_script_path)}"
    )
    healthcheck_command_by_idx[controller_seed] = f"tmux has-session -t {shlex.quote(queue_session_name)}"
    completion_command_by_idx[controller_seed] = f"test -f {shlex.quote(queue_done_marker)}"
    if follow_logs:
        follow_logs_command_by_idx[controller_seed] = (
            f"tail -n {shlex.quote(follow_log_lines)} -F {shlex.quote(queue_log_path)}"
        )

    babysit_idxs = [controller_seed]
    print(
        f'[run_picodo] sequential seed queue enabled for seeds={seed_idxs} in zone={controller_zone} '
        f'(region={controller_region}) on a single TPU VM.'
    )
else:
    for seed in seed_idxs:
        seed_zone = seed_zone_by_idx[seed]
        zones_by_idx[seed] = seed_zone
        seed_region = seed_zone.rsplit('-', 1)[0]

        if checkpoint_root:
            seed_checkpoint_root = checkpoint_root
        else:
            if seed_region not in checkpoint_root_by_region:
                fineweb_bucket = tn.get_fineweb_bucket_name(seed_zone, project_id)
                checkpoint_root_by_region[seed_region] = f'gs://{fineweb_bucket}/picodo_ckpts'
            seed_checkpoint_root = checkpoint_root_by_region[seed_region]

        env_exports = {
            'SEED': str(seed),
            'LR_TAG': lr_tag,
            'RUN_NAME_PREFIX': run_name_prefix,
            'CHECKPOINT_ROOT': seed_checkpoint_root,
            'BATCH_SIZE': batch_size,
            'NUM_TP_DEVICES': num_tp_devices,
            'WANDB_MODE': wandb_mode,
            'CONFIG_NAME': config_name,
            'CHECKPOINT_FREQ': checkpoint_freq,
            'EVAL_FREQ': eval_freq,
            'USE_CHINCHILLA': use_chinchilla,
            'USE_Z_LOSS': use_z_loss,
            'USE_B2_COSINE_ANNEAL': use_b2_cosine_anneal,
            'B2_ARG': b2_arg,
            'FINAL_B2_ARG': final_b2_arg,
            'WANDB_TOKEN': wandb_token,
        }
        if hf_token:
            env_exports['HF_TOKEN'] = hf_token
        if lr_arg:
            env_exports['LR_ARG'] = lr_arg

        exports_text = '\n'.join(
            f"export {key}={shlex.quote(value)}"
            for key, value in env_exports.items()
        )
        ssh_script_by_idx[seed] = (
            f"{exports_text}\n"
            f"{remote_runner_bootstrap}\n"
            f"bash {shlex.quote(remote_runner_script_path)}"
        )
        healthcheck_command_by_idx[seed] = f"tmux has-session -t {shlex.quote(f'picodo_train_seed{seed}')}"
        done_marker = (
            f"/home/tingchen/loss-spikes-project/picodo/train_status/"
            f"{run_name_prefix}_seed{seed}_lr{lr_tag}_bs{batch_size}.done"
        )
        completion_command_by_idx[seed] = f"test -f {shlex.quote(done_marker)}"

        if follow_logs:
            run_name = f"{run_name_prefix}_seed{seed}_lr{lr_tag}_bs{batch_size}"
            log_path = f"/home/tingchen/loss-spikes-project/picodo/train_logs/{run_name}.log"
            follow_logs_command_by_idx[seed] = (
                f"tail -n {shlex.quote(follow_log_lines)} -F {shlex.quote(log_path)}"
            )

tn.babysit(
    idxs=babysit_idxs,
    tpu_type=tpu_type,
    zone=default_zone,
    project_id=project_id,
    ssh_script=run_script,
    ssh_script_by_idx=ssh_script_by_idx,
    follow_logs_command_by_idx=follow_logs_command_by_idx,
    healthcheck_command_by_idx=healthcheck_command_by_idx,
    completion_command_by_idx=completion_command_by_idx,
    delete_on_completion=True,
    zones_by_idx=zones_by_idx,
    tpu_id_prefix=tpu_id_prefix,
    # ssh_script='cd loss-spikes-project/picodo && git pull',
    # ssh_script="pkill -9 python3 || true",
    startup_script=None,
    ensure_nat=True
)
