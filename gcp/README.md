# TP-MCTS on Google Compute Engine (64-core CPU)

Run the same workflow as `experiments.ipynb` on a Linux VM with many vCPUs—no Docker per experiment, no TPU. The VM clones your GitHub repo once; **Setup** with `RUN_ENV = "gce"` pulls updates and installs Python deps.

## Cost (trial $300)

| Item | Approximate |
|------|-------------|
| `n2-highcpu-64` (64 vCPU) | ~\$2.5–3.5 / hour (region-dependent) |
| 100 GB disk | ~\$10 / month while disk exists |

**Stop the VM when idle** (disk kept): you pay mostly for disk. **Delete the VM** when finished with a batch of runs.

```bash
gcloud compute instances stop tp-mcts-64cpu --zone=us-central1-a
# or delete:
gcloud compute instances delete tp-mcts-64cpu --zone=us-central1-a
```

## Prerequisites

1. [Google Cloud trial](https://cloud.google.com/free) with billing enabled.
2. Create a project; note **Project ID**.
3. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) on Windows.
4. Login and set project:

```powershell
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

5. Enable Compute Engine API (Console → APIs, or):

```powershell
gcloud services enable compute.googleapis.com
```

6. **Push your branch to GitHub** before relying on clone-on-VM (the VM uses the public repo URL in `config.env`).

## One-time: create the VM

**Windows (PowerShell):**

```powershell
cd gcp
copy config.env.example config.env   # edit PROJECT_ID if needed
..\gcp\create-vm.ps1
```

**Git Bash / WSL / Cloud Shell:**

```bash
cd gcp
cp config.env.example config.env
chmod +x create-vm.sh bootstrap-vm.sh startup.sh
./create-vm.sh
```

If `n2-highcpu-64` fails (quota or zone capacity), use `e2-highcpu-8` and another zone in `config.env` (see Quota section below).

On Windows you can use [Cloud Shell](https://console.cloud.google.com/cloudshell) (upload `gcp/` or clone your repo there) instead of local bash.

Check startup finished:

```bash
gcloud compute ssh tp-mcts-64cpu --zone=us-central1-a --command="test -f /opt/tp_mcts/.tp_mcts_gce && echo OK"
```

## Run experiments (same notebook flow)

### Option A — Cursor / VS Code Remote SSH (recommended)

1. `gcloud compute config-ssh` (adds SSH host alias).
2. Cursor: **Remote-SSH** → host like `tp-mcts-64cpu.us-central1-a.YOUR_PROJECT`.
3. Open folder `/opt/tp_mcts`.
4. Open `experiments.ipynb`, kernel **Python 3** (or **TP-MCTS** after bootstrap).
5. **Setup** cell: `RUN_ENV = "gce"` (or `"auto"` on this VM).
6. **Config** → run experiment cells. Output prints in the notebook as today.
7. Results live in `/opt/tp_mcts/results/`. Copy to PC:

```powershell
.\gcp\fetch-results.ps1
```

### Option B — Jupyter in browser via SSH tunnel

Terminal 1 (VM):

```bash
gcloud compute ssh tp-mcts-64cpu --zone=us-central1-a
cd /opt/tp_mcts
/opt/tp_mcts/.venv/bin/jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

Terminal 2 (Windows):

```powershell
.\gcp\ssh-jupyter.ps1
```

Open the URL Jupyter prints (`http://127.0.0.1:8888/...`). In the notebook: `RUN_ENV = "gce"`.

### Option C — Headless (no notebook)

```bash
gcloud compute ssh tp-mcts-64cpu --zone=us-central1-a
cd /opt/tp_mcts
export PYTHONPATH=/opt/tp_mcts
python3 scripts/run_mcts_heuristic_comparison.py --runs 1 --seed 123 \
  --heuristics baseline --objects 2 --deadlines 25 --verbose
```

## `RUN_ENV = "gce"` behavior (in `experiments.ipynb`)

| | Colab | GCE |
|---|--------|-----|
| Repo path | `/content/tp_mcts` | `/opt/tp_mcts` |
| Each Setup | Deletes & re-clones | `git pull` (keeps `results/`) |
| Results | Lost on disconnect | Persist on disk until you delete VM |

Set `REPO_BRANCH` in Setup when testing a feature branch (must exist on GitHub).

## Updating code on the VM

1. Push to GitHub.
2. Re-run **Setup** in the notebook (`RUN_ENV = "gce"`), or on the VM:

```bash
git -C /opt/tp_mcts pull
```

## Machine types

Default: `n2-highcpu-64` (64 vCPU, 64 GB RAM). Change `MACHINE_TYPE` in `config.env`, e.g.:

- `n2-highcpu-32` — cheaper smoke tests  
- `c3-highcpu-88` — newer generation, more vCPU (check quota & price)

Request quota increase in Console → IAM & admin → Quotas if create fails.

## Quota / billing tips

- **New trial projects** often have **`CPUS_ALL_REGIONS` = 12** globally. A 64-vCPU VM will fail until you [request a quota increase](https://console.cloud.google.com/iam-admin/quotas?project=YOUR_PROJECT_ID) for **Compute Engine API → CPUs (all regions)** (and optionally **N2 CPUs**). Ask for e.g. **128** CPUs for experiments.
- Until approved, use **`e2-highcpu-8`** (8 vCPU) or **`n2-highcpu-8`** in an available zone (`us-east1-d` worked for this project).
- Pick a zone close to you with available capacity (if create fails with `ZONE_RESOURCE_POOL_EXHAUSTED`, try another zone or machine family).
- Use **stop** not **delete** between sessions if you will resume the same disk.
- Trial credits apply to Compute; watch **Billing → Reports**.

### Request 64-core quota (one-time)

1. Console → **IAM & admin** → **Quotas**
2. Filter: **Compute Engine API**, metric **CPUs (all regions)**
3. **Edit quotas** → request **128** (or **96**)
4. After approval, set `MACHINE_TYPE=n2-highcpu-64` in `config.env`, delete the small VM, run `create-vm.ps1` again (or resize — recreating is simpler).

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Startup not done | `gcloud compute ssh ... --command='sudo tail -50 /var/log/tp-mcts-startup.log'` |
| `fixed_tail` missing in Config | Push branch; set `REPO_BRANCH` in Setup; re-run Setup |
| Slow single run | MCTS is mostly single-threaded per episode; 64 cores help parallel sweeps / OS headroom, not one tree |
| Cannot SSH | Firewall allows SSH by default; use `gcloud compute ssh` |

## Files

| File | Purpose |
|------|---------|
| `config.env.example` | Project, zone, machine type, repo URL |
| `create-vm.sh` | Create VM + startup script |
| `bootstrap-vm.sh` | Python, Jupyter, clone/pull repo |
| `fetch-results.ps1` | `scp` results to Windows |
| `ssh-jupyter.ps1` | Port-forward 8888 |
