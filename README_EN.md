# ModelGym

A desktop workspace for model training servers, based on RackTop 1.26.2. ModelGym retains server monitoring, SSH access, project synchronization and history, and adds remote GPU training queues.

[中文说明](README.md) · [Training guide](docs/MODEL_TRAINING.md)

Configure an existing remote training script and configuration file, choose Python, Conda or torchrun, and select fixed GPUs. Training starts only after all selected GPUs have no compute processes, utilization at or below 10%, enough free memory, and 30 seconds of stable availability. Busy GPUs require explicit queue confirmation.

The queue runs on the remote Linux server and survives desktop exit, sleep and SSH disconnection. View status and logs, cancel waiting jobs or stop a running training job. Overlapping GPU requests preserve submission order; disjoint GPU sets can run concurrently. Failed training is not automatically retried.

The initial training scheduler requires Python 3.8+ and NVIDIA physical GPUs with working nvidia-smi. It does not provide MIG scheduling, cross-server distributed training or cluster-wide hardware isolation.

Windows installers are delivered in the local `artifacts/windows/` directory. Exit the old RackTop process, including its tray instance, before upgrading. ModelGym preserves the original application identity, database, credentials and historical data. Upstream RackTop automatic updates are disabled for this distribution.

See [README.md](README.md) for development and build commands, and the [training guide](docs/MODEL_TRAINING.md) for usage and operational limits.

ModelGym is derived from [Tongzh-SEU/RackTop](https://github.com/Tongzh-SEU/RackTop), with upstream attribution and the original [GNU GPL v3 license](LICENSE) retained.
