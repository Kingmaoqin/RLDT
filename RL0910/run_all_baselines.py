#!/usr/bin/env python3
"""
Run multiple online baselines (using your modified online_loop.py that reads RL_BASELINE)
and collect PAPER COMPLIANCE REPORT metrics into a CSV.

Usage examples:
  python run_all_baselines.py
  python run_all_baselines.py --algos dqn,double_dqn,nfq,cql,calql,bcq --mode 1 --duration 100
  UPDATE_STEPS=20 UPDATE_FREQUENCY=20 BATCH_SIZE=32 python run_all_baselines.py --algos dqn,cql --mode 1 --duration 100
"""
import argparse, os, sys, subprocess, datetime, pathlib, re, csv

METRIC_KEYS = ["query_rate", "response_time", "throughput", "safety"]

def parse_report(stdout_text: str):
    """Parse stdout to extract metrics from PAPER COMPLIANCE REPORT and some extras."""
    data = {
        "overall_score": None,
        "tests_passed": None,
        "tests_total": None,
        "query_rate": None,
        "response_time": None,
        "throughput": None,
        "safety": None,
        "updates": None,
        "last_buffer": None,
        "algo_used": None,
        "fallback_to_cql": False,
        "return_code": 0,
    }

    # Overall score & Tests
    m = re.search(r"Overall Score:\s*([0-9.]+)%", stdout_text)
    if m: data["overall_score"] = float(m.group(1))

    m = re.search(r"Tests Passed:\s*(\d+)\/(\d+)", stdout_text)
    if m:
        data["tests_passed"] = int(m.group(1))
        data["tests_total"] = int(m.group(2))

    # Four metrics (accept both ✅ and ❌ lines)
    for key in METRIC_KEYS:
        pat = rf"(?:✅|❌)\s*{key}:\s*([0-9.]+)"
        m = re.search(pat, stdout_text, flags=re.IGNORECASE)
        if m:
            data[key] = float(m.group(1))

    # Updates & buffer (last occurrence wins)
    upd_matches = list(re.finditer(r"\[[A-Z_]+\]\s*updated:.*?updates=(\d+),\s*buffer=(\d+)", stdout_text))
    if upd_matches:
        last = upd_matches[-1]
        data["updates"] = int(last.group(1))
        data["last_buffer"] = int(last.group(2))
    else:
        # Older BCQ print format (if any)
        bcq_matches = list(re.finditer(r"BCQ updated:.*?updates=(\d+),\s*buffer=(\d+)", stdout_text))
        if bcq_matches:
            last = bcq_matches[-1]
            data["updates"] = int(last.group(1))
            data["last_buffer"] = int(last.group(2))

    # Algo actually used (from GenericTrainer banner) and fallback detection
    m = re.search(r"\[GenericTrainer\]\s*algo=([a-zA-Z0-9_]+)\s+device=", stdout_text)
    if m:
        data["algo_used"] = m.group(1).lower()

    if "calql package not found — fallback to CQL" in stdout_text:
        data["fallback_to_cql"] = True

    return data

def which_python():
    return sys.executable or "python"

def run_one(algo: str, mode: int, duration: int, logdir: pathlib.Path):
    """Run one baseline and return parsed metrics + raw log path."""
    env = os.environ.copy()
    env["RL_BASELINE"] = algo
    # Allow override via environment for batch size / steps / frequency
    # (user can export UPDATE_STEPS, UPDATE_FREQUENCY, BATCH_SIZE before running)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logdir / f"{algo}_{ts}.stdout.log"

    cmd = [which_python(), "run_complete_evaluation.py", "--mode", str(mode), "--duration", str(duration)]
    print(f"[runner] Algo={algo} | CMD: {' '.join(cmd)}")
    print(f"[runner] Logging to: {log_path}")

    # Stream output to both terminal and file
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        captured = []
        for line in iter(proc.stdout.readline, b""):
            try:
                s = line.decode("utf-8", errors="replace")
            except Exception:
                s = str(line)
            sys.stdout.write(s)
            logf.write(s)
            captured.append(s)
        ret = proc.wait()
        stdout_text = "".join(captured)
        metrics = parse_report(stdout_text)
        metrics["return_code"] = ret
    return metrics, log_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", type=str, default="dqn,double_dqn,nfq,cql,calql,bcq",
                    help="Comma-separated list: dqn,double_dqn,nfq,cql,calql,bcq (order matters)")
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--duration", type=int, default=100)
    ap.add_argument("--out", type=str, default="baseline_results.csv",
                    help="CSV output filename")
    ap.add_argument("--logdir", type=str, default="comparison_logs",
                    help="Directory to store per-run logs")
    args = ap.parse_args()

    algos = [a.strip().lower() for a in args.algos.split(",") if a.strip()]
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for algo in algos:
        metrics, log_path = run_one(algo, args.mode, args.duration, logdir)
        row = {
            "algo_requested": algo,
            "algo_used": metrics.get("algo_used") or algo,
            "fallback_to_cql": int(bool(metrics.get("fallback_to_cql"))),
            "overall_score": metrics.get("overall_score"),
            "tests_passed": metrics.get("tests_passed"),
            "tests_total": metrics.get("tests_total"),
            "query_rate": metrics.get("query_rate"),
            "response_time": metrics.get("response_time"),
            "throughput": metrics.get("throughput"),
            "safety": metrics.get("safety"),
            "updates": metrics.get("updates"),
            "last_buffer": metrics.get("last_buffer"),
            "return_code": metrics.get("return_code"),
            "log_path": str(log_path),
        }
        rows.append(row)

    # Write CSV
    out_path = pathlib.Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "algo_requested","algo_used","fallback_to_cql","overall_score","tests_passed","tests_total",
            "query_rate","response_time","throughput","safety","updates","last_buffer","return_code","log_path"
        ])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n[runner] Done. CSV saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
