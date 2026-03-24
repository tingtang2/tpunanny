import os
import time
import subprocess
import threading
from google.cloud import tpu_v2alpha1
from google.api_core.exceptions import NotFound

client = tpu_v2alpha1.TpuClient()
_stop_event = threading.Event()
_threads = []


def get_runtime(tpu_type):
    # https://cloud.google.com/tpu/docs/runtimes
    if 'v6e' in tpu_type: return 'v2-alpha-tpuv6e'
    elif 'v5p' in tpu_type: return 'v2-alpha-tpuv5'
    elif 'v5lite' in tpu_type: return 'v2-alpha-tpuv5-lite'
    else: return 'tpu-ubuntu2204-base'


def _region_from_zone(zone):
    parts = zone.rsplit('-', 1)
    if len(parts) != 2:
        raise ValueError(f'Invalid zone: {zone}')
    return parts[0]


def _run_gcloud(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _ensure_cloud_nat(zone, project_id, network='default'):
    """Creates a regional Cloud Router and NAT if they do not already exist."""
    region = _region_from_zone(zone)
    router_name = f'tpunanny-router-{network}-{region}'
    nat_name = f'tpunanny-nat-{network}-{region}'
    created_anything = False

    router_describe = _run_gcloud([
        'gcloud', 'compute', 'routers', 'describe', router_name,
        f'--region={region}',
        f'--project={project_id}',
        '--format=value(name)',
    ])
    if router_describe.returncode != 0:
        print(f'[nat] creating Cloud Router {router_name} in {region} on network {network}...')
        router_create = _run_gcloud([
            'gcloud', 'compute', 'routers', 'create', router_name,
            f'--network={network}',
            f'--region={region}',
            f'--project={project_id}',
            '--quiet',
        ])
        if router_create.returncode != 0:
            raise RuntimeError(
                f'Failed to create Cloud Router {router_name}: {router_create.stderr.strip() or router_create.stdout.strip()}'
            )
        created_anything = True

    nat_describe = _run_gcloud([
        'gcloud', 'compute', 'routers', 'nats', 'describe', nat_name,
        f'--router={router_name}',
        f'--region={region}',
        f'--project={project_id}',
        '--format=value(name)',
    ])
    if nat_describe.returncode != 0:
        print(f'[nat] creating Cloud NAT {nat_name} in {region}...')
        nat_create = _run_gcloud([
            'gcloud', 'compute', 'routers', 'nats', 'create', nat_name,
            f'--router={router_name}',
            f'--region={region}',
            '--nat-all-subnet-ip-ranges',
            '--auto-allocate-nat-external-ips',
            f'--project={project_id}',
            '--quiet',
        ])
        if nat_create.returncode != 0:
            raise RuntimeError(
                f'Failed to create Cloud NAT {nat_name}: {nat_create.stderr.strip() or nat_create.stdout.strip()}'
            )
        created_anything = True

    print(f'[nat] Cloud NAT ready: {nat_name} via router {router_name} in {region}.')
    return created_anything


def _create(tpu_id, tpu_type, zone, project_id, startup_script=None):
    parent = f'projects/{project_id}/locations/{zone}'

    node_metadata = {}
    if startup_script is not None:
        node_metadata['startup-script'] = startup_script

    queued_resource = tpu_v2alpha1.QueuedResource(
        tpu=tpu_v2alpha1.QueuedResource.Tpu(
            node_spec=[
                tpu_v2alpha1.QueuedResource.Tpu.NodeSpec(
                    parent=parent,
                    node_id=tpu_id,
                    node=tpu_v2alpha1.Node(
                        accelerator_type=tpu_type,
                        runtime_version=get_runtime(tpu_type),
                        network_config=tpu_v2alpha1.NetworkConfig(enable_external_ips=False),
                        metadata=node_metadata,
                    ),
                )
            ]
        ),
        spot=tpu_v2alpha1.QueuedResource.Spot(),
    )

    operation = client.create_queued_resource(
        parent=parent,
        queued_resource_id=tpu_id,
        queued_resource=queued_resource,
    )
    
    return operation.result()


def _delete(tpu_id, zone, project_id):
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    request = tpu_v2alpha1.DeleteQueuedResourceRequest(name=qr_name, force=True)
    operation = client.delete_queued_resource(request=request)
    return operation


def _delete_all_suspended(project_id):
    """
    Deletes all queued resources in SUSPENDED state across all zones.
    Returns a list of dicts with 'tpu_id' and 'zone' for each deleted resource.
    """

    deleted, pending = [], []
    queued_resources = client.list_queued_resources(parent=f'projects/{project_id}/locations/-')
    for qr in queued_resources:
        zone = qr.name.split('/')[3]
        qr_id = qr.name.split('/')[-1]
        state = qr.state.state.name
        if state in ('FAILED', 'SUSPENDED'):
            pending.append((_delete(qr_id, zone, project_id), qr_id, zone))

    for operation, qr_id, zone in pending:
        try:
            operation.result()
            deleted.append({'tpu_id': qr_id, 'zone': zone})
            print(f'Deleted suspended queued resource {qr_id} in {zone}')
        except NotFound:
            print(f'Warning: queued resource {qr_id} in {zone} not found (already deleted?)')
        except Exception as e:
            print(f'Error: failed to delete {qr_id} in {zone}: {e}')

    return deleted


def _recreate(tpu_id, tpu_type, zone, project_id, startup_script=None):
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    
    tpu_state = _get_tpu_state(qr_name)
    if tpu_state is None:
        # if TPU doesn't exist, create it
        _create(tpu_id, tpu_type, zone, project_id, startup_script)
        return 'created'

    print(f'[{tpu_id}] observed TPU state={tpu_state}')

    # if TPU is unhealthy, delete it and create a new one
    if tpu_state in ('FAILED', 'SUSPENDED', 'PREEMPTED'):
        _request_delete(tpu_id, zone, project_id)
        if not _wait_for_absence(qr_name):
            print(f'[{tpu_id}] waiting for deletion timed out; will retry.')
            return 'deleting'
        _create(tpu_id, tpu_type, zone, project_id, startup_script)
        return 're-created'

    # deletion in flight; wait for NotFound then recreate on next loop
    if tpu_state == 'DELETING':
        return 'deleting'

    return 'exists'


def _wait_for_absence(qr_name, timeout_seconds=300, poll_seconds=5):
    """Waits until queued resource is deleted, returns False on timeout."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            client.get_queued_resource(name=qr_name)
        except NotFound:
            return True
        time.sleep(poll_seconds)
    return False


def _get_tpu_state(qr_name):
    """Returns queued resource state name or None when resource is absent."""
    try:
        tpu_info = client.get_queued_resource(name=qr_name)
        return tpu_info.state.state.name
    except NotFound:
        return None


def _request_delete(tpu_id, zone, project_id):
    """
    Requests queued resource deletion without waiting on operation.result().
    This avoids occasional response type conversion issues in the client.
    """
    try:
        _delete(tpu_id, zone, project_id)
        return True
    except NotFound:
        return True
    except Exception as e:
        print(f'[{tpu_id}] delete request failed: {e}')
        return False


def _run(tpu_id, zone, project_id, ssh_script, log_prefix='ssh'):
    """Runs `ssh_script` on all workers of a TPU VM via gcloud SSH."""
    output_dir = os.path.join('logs', zone, tpu_id)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'gcloud', 'alpha', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_id,
        f'--zone={zone}',
        '--tunnel-through-iap',
        f'--project={project_id}',
        '--worker=all',
        f'--command={ssh_script}',
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_buffer = []
    stderr_buffer = []

    def _stream_pipe(stream, label, log_path, buffer):
        with open(log_path, 'w') as log_file:
            for line in stream:
                buffer.append(line)
                log_file.write(line)
                log_file.flush()
                print(f'[{tpu_id}] ssh script {label}: {line.rstrip()}')
        stream.close()

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, 'stdout', os.path.join(output_dir, f'{log_prefix}.stdout.log'), stdout_buffer),
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, 'stderr', os.path.join(output_dir, f'{log_prefix}.stderr.log'), stderr_buffer),
    )
    stdout_thread.start()
    stderr_thread.start()

    returncode = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    return subprocess.CompletedProcess(
        args=cmd,
        returncode=returncode,
        stdout=''.join(stdout_buffer),
        stderr=''.join(stderr_buffer),
    )


def _follow_logs(tpu_id, zone, project_id, stop_event, follow_logs_command, retry_seconds=10):
    """Continuously tails logs on TPU workers, reconnecting if SSH/IAP drops."""
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    while not stop_event.is_set():
        tpu_state = _get_tpu_state(qr_name)
        if tpu_state != 'ACTIVE':
            stop_event.wait(20)
            continue

        print(f'[{tpu_id}] starting follow-logs session...')
        result = _run(tpu_id, zone, project_id, follow_logs_command, log_prefix='follow_logs')
        print(f'[{tpu_id}] follow-logs session exited with code {result.returncode}.')
        if stop_event.is_set():
            break
        if result.returncode != 0:
            stderr_text = (result.stderr or '').upper()
            if any(token in stderr_text for token in ('DELETING', 'PREEMPTED', 'NOT_FOUND')):
                stop_event.wait(20)
                continue
        stop_event.wait(retry_seconds)


def _babysit(
    tpu_id,
    tpu_type,
    zone,
    project_id,
    stop_event,
    ssh_script=None,
    startup_script=None,
    follow_logs_command=None,
    healthcheck_command=None,
    completion_command=None,
    delete_on_completion=True,
):
    """(Re)creates TPU and runs `ssh_script`."""
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    ran_ssh_script = False
    follow_logs_thread = None

    print(f'[{tpu_id}] starting to babysit...')
    while not stop_event.is_set():

        # check if TPU is healthy
        print(f'[{tpu_id}] checking TPU status...')
        try:
            create_status = _recreate(tpu_id, tpu_type, zone, project_id, startup_script)
        except Exception as e:
            print(f'[{tpu_id}] recreate check failed: {e}')
            stop_event.wait(10)
            continue
        print(f'[{tpu_id}] TPU status: {create_status}')
        if create_status != 'exists': ran_ssh_script = False
        elif ran_ssh_script:
            if completion_command is not None:
                completion_result = _run(tpu_id, zone, project_id, completion_command, log_prefix='completioncheck')
                if completion_result.returncode == 0:
                    print(f'[{tpu_id}] completion check passed; training finished.')
                    if delete_on_completion:
                        print(f'[{tpu_id}] deleting TPU queued resource after successful completion...')
                        if _request_delete(tpu_id, zone, project_id):
                            if _wait_for_absence(qr_name, timeout_seconds=600, poll_seconds=10):
                                print(f'[{tpu_id}] TPU queued resource deleted.')
                            else:
                                print(f'[{tpu_id}] timed out waiting for queued resource deletion.')
                    break

            # Script was launched before; ensure the remote trainer is still alive.
            if healthcheck_command is not None:
                health_result = _run(tpu_id, zone, project_id, healthcheck_command, log_prefix='healthcheck')
                if health_result.returncode != 0:
                    print(f'[{tpu_id}] healthcheck failed; will relaunch ssh script.')
                    ran_ssh_script = False

        # if an SSH script was provided, wait until TPU is ready, then run it
        if not ran_ssh_script and ssh_script is not None:

            # wait until TPU is ready
            while not stop_event.is_set():
                tpu_state = _get_tpu_state(qr_name)
                state_text = tpu_state if tpu_state is not None else 'NOT_FOUND'
                print(f'[{tpu_id}] TPU state={state_text}')
                if tpu_state == 'ACTIVE':
                    break
                stop_event.wait(10)

            if stop_event.is_set(): break

            # run ssh script
            print(f'[{tpu_id}] running ssh script...')
            result = _run(tpu_id, zone, project_id, ssh_script)
            print(f'[{tpu_id}] ssh script finished with exit code {result.returncode}.')
            ran_ssh_script = True

            if follow_logs_command is not None and follow_logs_thread is None:
                print(f'[{tpu_id}] follow-logs mode enabled.')
                follow_logs_thread = threading.Thread(
                    target=_follow_logs,
                    args=(tpu_id, zone, project_id, stop_event, follow_logs_command),
                    daemon=True,
                )
                follow_logs_thread.start()

        # wait before checking on the TPU again
        print(f'[{tpu_id}] sleeping...')
        stop_event.wait(60)


def babysit(
    idxs,
    tpu_type,
    zone,
    project_id,
    ssh_script=None,
    startup_script=None,
    ensure_nat=True,
    network='default',
    zones_by_idx=None,
    ssh_script_by_idx=None,
    follow_logs_command=None,
    follow_logs_command_by_idx=None,
    healthcheck_command=None,
    healthcheck_command_by_idx=None,
    completion_command=None,
    completion_command_by_idx=None,
    delete_on_completion=True,
    tpu_id_prefix='tn',
):
    """Keeps multiple TPUs alive, optionally running per-index `ssh_script` and `startup_script` on boot."""
    global _stop_event, _threads

    # stop any previously running babysit threads
    _stop_event.set()
    for thread in _threads:
        thread.join(timeout=5)
    _threads = []
    _stop_event = threading.Event()

    zones_by_idx = zones_by_idx or {}
    ssh_script_by_idx = ssh_script_by_idx or {}
    follow_logs_command_by_idx = follow_logs_command_by_idx or {}
    healthcheck_command_by_idx = healthcheck_command_by_idx or {}
    completion_command_by_idx = completion_command_by_idx or {}
    zones_to_use = sorted({zones_by_idx.get(idx, zone) for idx in idxs})

    if ensure_nat:
        created_any_nat = False
        for zone_to_use in zones_to_use:
            created_nat = _ensure_cloud_nat(zone_to_use, project_id, network=network)
            created_any_nat = created_any_nat or created_nat
        if created_any_nat:
            print('[nat] waiting 60s for Cloud NAT propagation before creating TPUs...')
            _stop_event.wait(60)

    # create and start a thread for each TPU
    threads = []
    for idx in idxs:
        idx_zone = zones_by_idx.get(idx, zone)
        idx_ssh_script = ssh_script_by_idx.get(idx, ssh_script)
        idx_follow_logs_command = follow_logs_command_by_idx.get(idx, follow_logs_command)
        idx_healthcheck_command = healthcheck_command_by_idx.get(idx, healthcheck_command)
        idx_completion_command = completion_command_by_idx.get(idx, completion_command)
        tpu_id = f'{tpu_id_prefix}-{tpu_type}-{idx}'
        thread = threading.Thread(
            target=_babysit,
            args=(
                tpu_id,
                tpu_type,
                idx_zone,
                project_id,
                _stop_event,
                idx_ssh_script,
                startup_script,
                idx_follow_logs_command,
                idx_healthcheck_command,
                idx_completion_command,
                delete_on_completion,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        _stop_event.wait(1) # stagger creation

    # keep main thread alive while threads are running
    try:
        _threads = threads
        while any(t.is_alive() for t in threads):
            _stop_event.wait(1)
    finally:
        _stop_event.set()
        for thread in threads:
            thread.join(timeout=5)
