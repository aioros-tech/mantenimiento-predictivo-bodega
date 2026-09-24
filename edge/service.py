# -*- coding: utf-8 -*-
"""
service.py — runs the same pipeline as src/reporte_slm.py on the Jetson
(TimesFM 2.5 -> facts computed in code -> the SLM writes the report), in a
loop, with both models loaded onto the GPU only once. Serves the dashboard
(src/construir_dashboard.py) over HTTP. See docs/DEPLOY-JETSON.md.

Input: a replay of data/anomaly-free.csv. Every cycle advances each unit by
`--step` rows (10 rows x 30 s = 5 min of stream) and forecasts again; when a
unit reaches the end of the CSV it wraps back to the start.

Every cycle writes to --output:
    datos_dashboard.json   same schema as src/generar_datos_dashboard.py
    dashboard/index.html   rebuilt with src/construir_dashboard.py
    reports.jsonl          one line per unit and cycle (facts + report)
    state.json             replay position (survives restarts)

Usage (inside a release):
    python edge/service.py --models-dir /opt/aioros-bodega/models --output /tmp/x --cycles 1
"""
import argparse, json, os, resource, shutil, subprocess, sys, threading, time, warnings
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # release root: edge/ src/ data/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "edge"))
# torch warns that its wheel has no sm_87 (Orin) kernels; it runs the sm_80 ones
# (binary compatible within 8.x, verified in docs/DEPLOY-JETSON.md §8).
warnings.filterwarnings("ignore", message=".*compute capability.*")


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def write_atomic(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def serve(directory, port):
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    srv = ThreadingHTTPServer(("0.0.0.0", port), partial(QuietHandler, directory=str(directory)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"dashboard at http://0.0.0.0:{port}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", default=str(ROOT / "edge/units.yaml"))
    ap.add_argument("--models-yaml", default=str(ROOT / "edge/models.yaml"))
    ap.add_argument("--models-dir", required=True, help="where download_models.py put the weights")
    ap.add_argument("--csv", default=str(ROOT / "data/anomaly-free.csv"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--interval", type=float, default=300, help="seconds between cycle starts")
    ap.add_argument("--step", type=int, default=10, help="CSV rows the replay advances per cycle")
    ap.add_argument("--port", type=int, default=0, help="serve the dashboard (0 = off)")
    ap.add_argument("--cycles", type=int, default=0, help="0 = run forever")
    ap.add_argument("--max-units", type=int, default=0, help="only the first N units (smoke test)")
    ap.add_argument("--require-cuda", action="store_true", help="abort if there is no GPU")
    a = ap.parse_args()

    # Local weights pinned by revision, set before importing reporte_slm (it reads the environment on import).
    from download_models import model_dir
    models = yaml.safe_load(open(a.models_yaml))
    for key, var in (("timesfm", "TFM_MODEL"), ("slm", "SLM_MODELO")):
        d = model_dir(a.models_dir, models[key])
        if not (d / ".download-complete").exists():
            sys.exit(f"ERROR: {key} weights missing in {d} (run edge/download_models.py)")
        os.environ[var] = str(d)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    import pandas as pd, torch
    import reporte_slm as rs
    from generar_datos_dashboard import build_unit_entry

    if a.require_cuda and rs.DISPOSITIVO != "cuda":
        sys.exit(f"ERROR: --require-cuda set but the device is {rs.DISPOSITIVO}")

    units = yaml.safe_load(open(a.units))["units"]
    if a.max_units:
        units = units[:a.max_units]
    df = pd.read_csv(a.csv, sep=";")
    min_rows = rs.CONTEXTO + rs.HORIZONTE       # what extraer_hechos needs (backtest)

    out = Path(a.output)
    (out / "dashboard").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "src/plantilla_dashboard.html", out / "plantilla_dashboard.html")
    state_path = out / "state.json"
    try:
        cursors = json.loads(state_path.read_text())
    except (OSError, ValueError):
        cursors = {}
    for u in units:
        cursors.setdefault(u["name"], u["start"])

    t0 = time.time()
    log(f"loading models on {rs.DISPOSITIVO} ({torch.cuda.get_device_name(0) if rs.DISPOSITIVO == 'cuda' else 'CPU'})")
    rs._modelo_tfm()
    mid, tok, mod = rs.cargar_slm()
    slm_name = models["slm"]["repo"] if mid == os.environ["SLM_MODELO"] else mid
    gpu_mb = torch.cuda.memory_allocated() / 2**20 if rs.DISPOSITIVO == "cuda" else 0
    log(f"models ready in {time.time() - t0:.1f}s: TimesFM 2.5 + {slm_name} | "
        f"GPU {gpu_mb:.0f} MB | peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")

    if a.port:
        serve(out / "dashboard", a.port)

    cycle = 0
    while True:
        cycle += 1
        cycle_start = time.time()
        data = {"equipos": [], "generado": time.strftime("%Y-%m-%d %H:%M")}   # dashboard schema
        for u in units:
            row = cursors[u["name"]]
            if row > len(df):
                row = min_rows
            series = df[u["column"]].astype("float32").to_numpy()[:row]
            r = build_unit_entry(u["name"], u["column"], series, float(u["limit"]), tok, mod)
            data["equipos"].append(r)
            with open(out / "reports.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "generated_at": data["generado"], "cycle": cycle, "row": row,
                    "unit": r["nombre"], "sensor": r["sensor"], "status": r["estado"],
                    "facts": r["hechos"], "report": r["reporte"], "guardrail_flags": r["guardarrail"],
                    "t_forecast_s": r["t_pronostico"], "t_slm_s": r["t_slm"],
                }, ensure_ascii=False) + "\n")
            flag = "" if not r["guardarrail"] else f" | GUARDRAIL: unverified figures {r['guardarrail']}"
            log(f"  {u['name']:24s} row {row:5d} {r['estado']:6s} "
                f"TimesFM {r['t_pronostico']:.2f}s + SLM {r['t_slm']:.1f}s{flag}")
            cursors[u["name"]] = row + a.step

        data["modelo_slm"] = slm_name
        data["dispositivo"] = rs.DISPOSITIVO
        write_atomic(out / "datos_dashboard.json", json.dumps(data, ensure_ascii=False, indent=1))
        subprocess.run([sys.executable, str(ROOT / "src/construir_dashboard.py")],
                       cwd=out, check=True, stdout=subprocess.DEVNULL)
        write_atomic(state_path, json.dumps(cursors, ensure_ascii=False))
        took = time.time() - cycle_start
        n_alert = sum(x["estado"] == "ALERTA" for x in data["equipos"])
        log(f"cycle {cycle}: {len(units)} units, {n_alert} in alert, {took:.1f}s")

        if a.cycles and cycle >= a.cycles:
            break
        if took > a.interval:
            log(f"  !! cycle took longer than --interval ({a.interval:.0f}s)")
        time.sleep(max(0.0, a.interval - took))


if __name__ == "__main__":
    main()
