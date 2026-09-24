# Deploying to the Jetson Orin Nano

How to take this repository's pipeline (TimesFM 2.5 forecasts → the code
extracts the facts → an SLM writes the report) to the Jetson Orin Nano, run it
there as a service, and serve the dashboard from the board.

It follows the same approach as the AIOROS Beta deployment
(`aioros-beta-ai/scripts/deploy-edge.sh`): immutable, versioned releases, a
`current` symlink that only moves after verification, a systemd service, and a
rollback script.

> **What runs today:** the same 4 units as the demo, with **SKAB** data
> (`data/anomaly-free.csv`) replayed as if it were a live stream. The SLM is
> **Qwen2.5-1.5B-Instruct**, the open fallback `reporte_slm.py` already uses,
> because Gemma is gated. The models run **unquantized**, in fp32 (TimesFM) and
> fp16 (Qwen), on the board's GPU.

---

## 1. Architecture

```
 laptop                                  Jetson Orin Nano (raisga@192.168.100.2)
 ──────                                  ───────────────────────────────────────
 src/  edge/  data/anomaly-free.csv      /opt/aioros-bodega/
        │                                  ├── venv/                   torch (CUDA) + timesfm + transformers, pinned
        └── scripts/deploy-edge.sh  ─────► ├── models/<repo>@<rev>/    weights pinned by revision (edge/models.yaml)
            (rsync + ssh)                  ├── releases/<release>/     immutable: edge/ src/ data/ SHA256SUMS
                                           │                           .aioros-release-ready (only after the smoke test)
                                           ├── current -> releases/<release>
                                           ├── output/                 datos_dashboard.json, dashboard/, reports.jsonl, state.json
                                           └── activation-history.tsv
                                         systemd: aioros-bodega-forecast.service  →  http://<jetson>:8095/
```

The service (`edge/service.py`) loads TimesFM and the SLM onto the GPU **once**
and then repeats a cycle:

1. For each unit in `edge/units.yaml`, it takes the series up to the replay's current row.
2. It calls `build_unit_entry()` from `src/generar_datos_dashboard.py`: TimesFM forecasts 32 min, the code extracts the facts and the SLM writes the report. It is the same function that produces the published dashboard, so the schema is the same.
3. It writes `datos_dashboard.json` and rebuilds `dashboard/index.html` with `src/construir_dashboard.py`, and appends one line per unit to `reports.jsonl`.
4. It advances the replay `--step` rows (10 rows × 30 s = 5 min) and waits until `--interval` (300 s) has passed, so the stream runs in real time. When a unit reaches the end of the CSV it wraps back to the start.

The first cycle starts at the same rows as the published dashboard and reproduces it (see §8.2).

## 2. Requirements

| Where | What |
|---|---|
| Jetson | JetPack 7 / L4T R39 (Ubuntu 24.04, **Python 3.12**, CUDA 13), internet access the first time (pip + Hugging Face), SSH key access for `raisga`. About 9 GB free under `/opt` (venv ≈ 5 GB, weights ≈ 3.8 GB). |
| Laptop | `rsync`, `ssh`. No Python needed: nothing is trained or exported on the laptop. |
| Network | The Jetson answers at `192.168.100.2` over the direct cable. If that link stalls, see the network troubleshooting entry in `aioros-beta-ai/docs/DEPLOY-JETSON.md` (EEE). |

### PyTorch on Orin with JetPack 7

There is no torch wheel built for **sm_87** (Orin) with CUDA 13. The PyPI,
pytorch.org and Jetson AI Lab (`sbsa/cu130`) wheels ship sm_80/90/100/110/120,
and torch warns at start-up that the GPU is "not supported". Thanks to CUDA
binary compatibility the sm_80 kernels **do run** on 8.7, and this was
verified with this model (§8.1).

`edge/requirements-edge.txt` pins the **PyPI aarch64 wheel** by URL and
sha256; it brings the CUDA 13 runtime (cuDNN included) as `nvidia-*` packages.
The Jetson AI Lab wheel expects a system cuDNN, which JetPack 7 does not ship:
the first deploy failed that way (`ImportError: libcudnn.so.9`). `service.py`
silences the sm_87 warning.

## 3. Step 1: one-time Jetson setup (asks for the password)

`/opt` belongs to root and the deploy never asks for a password, so a one-time
bootstrap prepares the board. Run it **in an interactive terminal** (the `!`
prefix in Claude Code has no TTY, so `sudo` can't prompt there):

```bash
./scripts/bootstrap-jetson.sh raisga@192.168.100.2
```

It creates `/opt/aioros-bodega` (owned by `raisga`), installs and enables
`aioros-bodega-forecast.service`, and writes `/etc/sudoers.d/aioros-bodega`,
which allows **only** `systemctl start|stop|restart|is-active` of that service
without a password. It also stops and removes the unit's previous name
(`aioros-bodega-pronostico.service`) if present. It is idempotent, and must be
re-run whenever the `.service` file changes.

## 4. Step 2: deploy

```bash
./scripts/deploy-edge.sh raisga@192.168.100.2
```

| # | Step | Detail |
|---|---|---|
| 1 | Local check | `edge/`, `src/` and `data/anomaly-free.csv` are present |
| 2 | Copy | New **immutable** `/opt/aioros-bodega/releases/<UTC>-<pid>/` (an existing name is rejected), plus `SHA256SUMS` |
| 3 | Runtime | Rebuilds `venv/` from scratch whenever the sha256 of `requirements-edge.txt` changes (pip does not replace a same-version package whose URL changed); otherwise a no-op. First time ≈ 3 min for torch |
| 4 | Weights | `edge/download_models.py` fetches TimesFM and Qwen into `models/<repo>@<revision>/` if missing (3.8 GB the first time). The service loads them **offline** (`HF_HUB_OFFLINE=1`) |
| 5 | Verify | **Stops the running service** (two copies of the models don't fit in 8 GB next to Alpha/Beta) and runs **one real cycle** on the Jetson: 1 unit, local weights, CUDA required, non-empty report, dashboard built. On success it writes `.aioros-release-ready`; on failure it starts the previous service again |
| 6 | Activate | `current -> releases/<release>`, entry in `activation-history.tsv`, restart, then waits (up to 3 min) for the journal to say **"models ready"**: `is-active` only says the process started. If the models don't load, it **restores the previous release** |

Options: `--release NAME` and `--no-activate` (prepare and verify without switching).

## 5. Step 3: use it

- **Dashboard:** `http://192.168.100.2:8095/` (or the Jetson's Wi-Fi IP). Rebuilt every cycle; reload the page. Port 8090 is taken by AgentDVR.
- **Logs:** `ssh raisga@192.168.100.2 journalctl -u aioros-bodega-forecast -f` (one line per unit and cycle, plus the guardrail warning when it fires).
- **Reports:** `ssh raisga@192.168.100.2 tail -f /opt/aioros-bodega/output/reports.jsonl`.
- **Restart:** `ssh raisga@192.168.100.2 sudo -n systemctl restart aioros-bodega-forecast`.

To change the pace, create `/opt/aioros-bodega/service.env` with
`AIOROS_BODEGA_INTERVAL=<s>`, `AIOROS_BODEGA_STEP=<rows>` or
`AIOROS_BODEGA_PORT=<n>` and restart. `state.json` keeps the replay position;
delete it to start over.

**Run it by hand** (one cycle, no service; stop the service first, see §8.3):

```bash
ssh raisga@192.168.100.2 'cd /opt/aioros-bodega/current && /opt/aioros-bodega/venv/bin/python \
  edge/service.py --models-dir /opt/aioros-bodega/models --output /tmp/trial --cycles 1 --interval 0'
```

## 6. Rollback

```bash
./scripts/rollback-edge.sh raisga@192.168.100.2              # list (* = current)
./scripts/rollback-edge.sh raisga@192.168.100.2 <release>    # switch + restart
```

Only releases that passed verification can be activated. Old weights stay in
`models/` (the folder name carries the revision), so a rollback needs no network.

## 7. Changes to `src/`

Only lines added for this deployment; the existing code is untouched.

- `reporte_slm.py`: `TFM_MODEL` (environment variable, defaults to the Hugging Face id) lets TimesFM load from a local folder. `generate_text()` writes the report with an already-loaded SLM (`redactar()` uses it). `cargar_slm()` now prints why a model failed to load instead of always blaming Gemma's licence.
- `generar_datos_dashboard.py`: the per-unit work moved into `build_unit_entry()` and the script body is under `if __name__ == "__main__"`. Running the script works as before.

## 8. Verification and results (2026-09-21)

### 8.1 GPU versus CPU on the Jetson

TimesFM 2.5, 5 windows of the CSV, same process with and without CUDA: largest
quantile difference **7.6e-6** (relative **8.6e-8**). The sm_80 kernels give the
same result as the CPU.

### 8.2 Jetson versus the published dashboard (Mac M5, MPS)

The Jetson's first cycle for the 4 units compared with `data/datos_dashboard.json`:

| | Result |
|---|---|
| Forecast and intervals (64 steps × 3 curves) | identical at the published rounding (max Δ 0.000) |
| Numeric facts (15 fields per unit) | **identical** for all 4 |
| NORMAL / ALERTA status | identical (1 in alert: Blast freezer 1) |
| SLM text | word-for-word identical for 2 of 4; different wording for the other 2 (fp16 on other hardware) |
| Numeric guardrail | OK for all 4 |

### 8.3 Performance and memory (Orin Nano 8 GB, 15 W mode, with Alpha and Beta running)

| | |
|---|---|
| Model loading | 15 s (from NVMe, weights already on disk) |
| TimesFM 2.5, one forecast | 0.36 s on GPU (0.61 s on CPU); the first one per process 0.9 s |
| SLM writing | 25–28 s per unit |
| 4-unit cycle | **111 s**, well within the 300 s interval |
| GPU memory | **3.8 GB** (Qwen fp16 2.9 GB + TimesFM fp32 0.9 GB) |
| System MemAvailable | 5.2 GB before, **≈ 0.5 GB** minimum while the service is generating |
| Board power | 10.3 W average during a cycle, 5.3 W idle, 11.2 W peak; max 62 °C |

**Memory is the tight spot.** The board has 7.3 GB shared between CPU and GPU,
and GPU-resident weights can't be swapped. There is headroom thanks to the 8 GB
swapfile and Alpha/Beta's small footprint, but not much. Therefore:

- the unit sets `OOMScoreAdjust=800`: if memory runs out, the kernel kills this service before Alpha/Beta, and systemd restarts it;
- the deploy's smoke test stops the service before loading a second copy.

## 9. Limitations and next steps

1. **Quantize the SLM** (the README anticipates it): Qwen2.5-1.5B in int4/int8 would drop from 2.9 GB to ≈ 1 GB and write faster. It needs a runtime supported on JetPack 7 (llama.cpp/GGUF or TensorRT-LLM), not transformers. The guardrail must be re-validated with the quantized model.
2. **Live input:** replace the replay with real telemetry. `service.py` only needs each unit's series up to "now".
3. **Gemma:** with a Hugging Face token and the licence accepted, change `slm` in `edge/models.yaml` and redeploy.
4. **SLM text:** in 5 of the first 7 ALERTA reports the text says the unit is in its normal range or at no risk, even though the forecast crosses the limit. It happens on the Mac too; part of the cause is the existing `hechos_en_palabras()`. The status and table, computed by the code, are correct. Details and fix in the code review report (`docs/reports/`).
5. The dashboard is served over plain HTTP on the local network, with no authentication.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `test -w /opt/aioros-bodega/releases` fails | Bootstrap not run → §3 |
| `sudo: a password is required` during activation | The sudoers rule is missing or the unit changed → re-run the bootstrap |
| `sudo: a terminal is required to read the password` in the bootstrap | Run it in a real terminal, not with the `!` prefix |
| `ERROR: … weights missing in …` | `models/` is incomplete → the deploy fetches it; by hand: `edge/download_models.py edge/models.yaml /opt/aioros-bodega/models` |
| `ERROR: --require-cuda set but the device is cpu` | torch without CUDA in the venv → check the `torch @` line of `requirements-edge.txt` |
| The service keeps restarting | `journalctl -u aioros-bodega-forecast -n 80`; look for `Killed`/OOM in `journalctl -k` → §8.3 |
| The dashboard doesn't load | Port 8095 taken? `ss -ltn`; change `AIOROS_BODEGA_PORT` |

## 11. Deployment record (2026-09-21)

| | |
|---|---|
| Board | Jetson Orin Nano `raisga@192.168.100.2`, L4T R39.2, CUDA 13.2, Python 3.12.3, 15 W mode |
| Active release | `/opt/aioros-bodega/releases/20260921T234307Z-498940` (repo `a86dd00` + the changes in this guide) |
| Runtime | torch 2.11.0+cu130 (PyPI aarch64 wheel), timesfm 3.0.2, transformers 5.17.0 |
| Weights | TimesFM 2.5 `1d952420…`, Qwen2.5-1.5B-Instruct `989aa798…` in `models/` |
| Service | `aioros-bodega-forecast.service` enabled and running; dashboard on `:8095` |
| Verified | smoke test on every deploy; full 4-unit cycle (111–115 s) identical to the published dashboard; dashboard over HTTP; deploy over the running service (stop → test → activate); rollback to the previous release and back; replay wrap-around at the end of the CSV; Beta stayed up throughout, no OOM |

The first two attempts failed at step 3, before anything was activated, and
the unverified releases were deleted:

1. The Jetson AI Lab wheel: `ImportError: libcudnn.so.9` (§2).
2. With the pin fixed, pip **did not replace** the torch already installed (same version, different URL) and kept loading the old binary (`libnvpl_lapack_lp64_gomp.so.0`). The deploy now rebuilds the venv from scratch when the sha256 of `requirements-edge.txt` changes, and the venv's `libtorch_cuda.so` was checked to be identical to the validated one.

Later the same day the deployment code was rewritten in English (file names,
flags, config keys, unit name `aioros-bodega-pronostico` → `aioros-bodega-forecast`,
`modelos/` → `models/`, `salida/` → `output/`). That required re-running the
bootstrap; the releases with the old names were removed because the new unit
cannot run them.

## 12. Files

| File | Role |
|---|---|
| `edge/service.py` | On-board runtime: models on the GPU → per-unit cycle → dashboard + reports |
| `edge/units.yaml` | Units, sensor column, limit and replay starting row |
| `edge/models.yaml` | Exact repos and revisions of TimesFM and the SLM |
| `edge/download_models.py` | Downloads the pinned weights into `models/` |
| `edge/requirements-edge.txt` | Exact runtime pins (torch by URL + sha256) |
| `edge/aioros-bodega-forecast.service` | systemd unit (always runs `/opt/aioros-bodega/current`) |
| `scripts/bootstrap-jetson.sh` | One-time setup with sudo |
| `scripts/deploy-edge.sh` | Release → runtime → weights → verification → activation (restores on failure) |
| `scripts/rollback-edge.sh` | List / switch releases |

### Client reports (`docs/reports/`)

Standalone HTML, open in any browser. The Spanish and English versions have the same content.

| File | Content |
|---|---|
| `bodega-prueba-borde.es.html` / `bodega-edge-test.en.html` | How it was tested, Mac versus Jetson, speed, memory and power, technical appendix |
| `bodega-revision-codigo.es.html` / `bodega-code-review.en.html` | What changed in `src/` and why, the deployment module, 7 findings by priority and a plan |
