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
    
    try:
        # get TPU status
        tpu_info = client.get_queued_resource(name=qr_name)
        tpu_state = tpu_info.state.state.name
        
        # if TPU is unhealthy, delete it and create a new one
        if tpu_state in ('FAILED', 'SUSPENDED'):
            _delete(tpu_id, zone, project_id).result()
            if not _wait_for_absence(qr_name):
                print(f'[{tpu_id}] waiting for deletion timed out; will retry.')
                return 'deleting'
            _create(tpu_id, tpu_type, zone, project_id, startup_script)
            return 're-created'

    except NotFound as e:
        # if TPU doesn't exist, create it
        _create(tpu_id, tpu_type, zone, project_id, startup_script)
        return 'created'

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


def _run(tpu_id, zone, project_id, ssh_script):
    """Runs `ssh_script` on all workers of a TPU VM via gcloud SSH."""
    import os
    output_dir = f'logs/{zone}/{tpu_id}'
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', tpu_id,
        f'--zone={zone}',
        f'--project={project_id}',
        '--worker=all',
        f'--command={ssh_script}',
        f'--output-directory={output_dir}',
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _babysit(tpu_id, tpu_type, zone, project_id, stop_event, ssh_script=None, startup_script=None):
    """(Re)creates TPU and runs `ssh_script`."""
    qr_name = f'projects/{project_id}/locations/{zone}/queuedResources/{tpu_id}'
    ran_ssh_script = False

    print(f'[{tpu_id}] starting to babysit...')
    while not stop_event.is_set():

        # check if TPU is healthy
        print(f'[{tpu_id}] checking TPU status...')
        create_status = _recreate(tpu_id, tpu_type, zone, project_id, startup_script)
        print(f'[{tpu_id}] TPU status: {create_status}')
        if create_status != 'exists': ran_ssh_script = False

        # if an SSH script was provided, wait until TPU is ready, then run it
        if not ran_ssh_script and ssh_script is not None:

            # wait until TPU is ready
            while not stop_event.is_set():
                tpu_info = client.get_queued_resource(name=qr_name)
                tpu_state = tpu_info.state.state.name
                print(f'[{tpu_id}] TPU state={tpu_state}')
                if tpu_state == 'ACTIVE': break
                stop_event.wait(10)

            if stop_event.is_set(): break

            # run ssh script
            print(f'[{tpu_id}] running ssh script...')
            result = _run(tpu_id, zone, project_id, ssh_script)
            print(f'[{tpu_id}] ssh script finished with exit code {result.returncode}.')
            ran_ssh_script = True

        # wait before checking on the TPU again
        print(f'[{tpu_id}] sleeping...')
        stop_event.wait(60)


def babysit(idxs, tpu_type, zone, project_id, ssh_script=None, startup_script=None, ensure_nat=True, network='default'):
    """Keeps multiple TPUs alive, optionally running `ssh_script` and `startup_script` on boot."""
    global _stop_event, _threads

    # stop any previously running babysit threads
    _stop_event.set()
    for thread in _threads:
        thread.join(timeout=5)
    _threads = []
    _stop_event = threading.Event()

    if ensure_nat:
        created_nat = _ensure_cloud_nat(zone, project_id, network=network)
        if created_nat:
            print('[nat] waiting 60s for Cloud NAT propagation before creating TPUs...')
            _stop_event.wait(60)

    # create and start a thread for each TPU
    threads = []
    for idx in idxs:
        tpu_id = f'tn-{tpu_type}-{idx}'
        thread = threading.Thread(
            target=_babysit,
            args=(tpu_id, tpu_type, zone, project_id, _stop_event, ssh_script, startup_script),
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
