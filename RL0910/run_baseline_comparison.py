
#!/usr/bin/env python3
import argparse, os, sys, subprocess, tempfile, pathlib, datetime

SITECUSTOMIZE = '\n# Auto patcher injected via PYTHONPATH to replace online_loop.BCQOnlineTrainer\nimport importlib, importlib.abc, importlib.machinery, importlib.util, sys\n\nclass _LoaderWrapper(importlib.abc.Loader):\n    def __init__(self, orig_loader):\n        self._orig = orig_loader\n\n    def create_module(self, spec):\n        if hasattr(self._orig, "create_module"):\n            return self._orig.create_module(spec)\n        return None\n\n    def exec_module(self, module):\n        # Let the original loader load the module first\n        self._orig.exec_module(module)\n        try:\n            from generic_trainer import GenericD3RLPyOnlineTrainer\n            # Monkey-patch online_loop.BCQOnlineTrainer to our generic trainer\n            setattr(module, "BCQOnlineTrainer", GenericD3RLPyOnlineTrainer)\n            print("[PATCH] online_loop.BCQOnlineTrainer -> GenericD3RLPyOnlineTrainer (via sitecustomize)")\n        except Exception as e:\n            print(f"[PATCH] Failed to patch online_loop: {e}")\n\nclass _OnlineLoopPatcher(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path, target=None):\n        if fullname == "online_loop":\n            # Find the real spec first\n            real_spec = importlib.machinery.PathFinder.find_spec(fullname, path)\n            if real_spec and real_spec.loader:\n                # Wrap loader\n                return importlib.util.spec_from_loader(fullname, _LoaderWrapper(real_spec.loader), origin=real_spec.origin)\n        return None\n\n# Install our finder at highest priority\nif not any(isinstance(f, _OnlineLoopPatcher) for f in sys.meta_path):\n    sys.meta_path.insert(0, _OnlineLoopPatcher())\n'
GENERIC_TRAINER = '\nimport os, time, threading, random\nfrom collections import deque\nimport numpy as np\n\n# d3rlpy is required by the user\'s project\nfrom d3rlpy.dataset import MDPDataset\nfrom d3rlpy.algos import DQNConfig, DoubleDQNConfig, NFQConfig\ntry:\n    # Different d3rlpy versions name this class differently\n    from d3rlpy.algos import DiscreteCQLConfig as _CQLConfig\nexcept Exception:\n    from d3rlpy.algos import CQLConfig as _CQLConfig\n\n# Optional Cal-QL package (not required). If absent, we fallback to CQL.\n_HAS_CALQL = False\ntry:\n    # hypothetical third-party package example\n    from calql import CalQL as _CalQLAlgo  # type: ignore\n    _HAS_CALQL = True\nexcept Exception:\n    _HAS_CALQL = False\n\ndef _env_flag(name, default):\n    v = os.getenv(name, str(default))\n    try:\n        return type(default)(v)\n    except Exception:\n        return default\n\nclass GenericD3RLPyOnlineTrainer:\n    \\"\\"\\"\n    A generic online trainer with the SAME interface your code expects for BCQOnlineTrainer:\n      - add_transition(transition) -> dict\n      - perform_update() -> dict\n      - predict(state: np.ndarray) -> int\n      - get_statistics() -> dict\n\n    It instantiates one of: CalQL (if available) / CQL / DQN / DoubleDQN / NFQ,\n    selected by env var COMPARISON_ALGO in {\'calql\',\'dqn\',\'double_dqn\',\'nfq\'}.\n    \\"\\"\\"\n    def __init__(self, bcq_policy_path: str, device: str = "cuda"):\n        self.device = device\n\n        algo_name = os.getenv("COMPARISON_ALGO", "calql").strip().lower()\n        self._algo_name = algo_name\n\n        # Hyperparams\n        self._batch_size = _env_flag("BCQ_BATCH_SIZE", _env_flag("BATCH_SIZE", 32))\n        self._update_frequency = _env_flag("BCQ_UPDATE_FREQUENCY", _env_flag("UPDATE_FREQUENCY", 20))\n        self._update_steps = _env_flag("BCQ_UPDATE_STEPS", _env_flag("UPDATE_STEPS", 20))\n\n        # Buffers & stats\n        self.labeled_buffer = []\n        self.online_buffer = self.labeled_buffer  # alias expected by caller\n        self._buffer_lock = threading.Lock()\n        self._built = False\n        self._update_count = 0\n        self.training_losses = deque(maxlen=200)\n        self._last_update_walltime = time.time()\n\n        # Create algorithm\n        self.algo = self._build_algo(algo_name, device=self.device)\n        print(f"[GenericTrainer] Using algo={self._algo_name} device={self.device} "\n              f"(batch={self._batch_size}, freq={self._update_frequency}, steps={self._update_steps})")\n\n    # ---- properties expected by caller ----\n    @property\n    def batch_size(self): return int(self._batch_size)\n    @batch_size.setter\n    def batch_size(self, v): \n        try: self._batch_size = max(1,int(v))\n        except: pass\n\n    @property\n    def update_frequency(self): return int(self._update_frequency)\n    @update_frequency.setter\n    def update_frequency(self, v):\n        try: self._update_frequency = max(1,int(v))\n        except: pass\n\n    @property\n    def update_count(self): return int(self._update_count)\n\n    # ---- algorithm factory ----\n    def _build_algo(self, name, device="cuda"):\n        name = name.lower()\n        if name == "dqn":\n            return DQNConfig().create(device=device)\n        elif name == "double_dqn":\n            return DoubleDQNConfig().create(device=device)\n        elif name == "nfq":\n            return NFQConfig().create(device=device)\n        elif name == "calql":\n            if _HAS_CALQL:\n                # True CalQL implementation available\n                return _CalQLAlgo(device=device)\n            # Fallback to (Discrete)CQL in d3rlpy as a conservative Q baseline\n            print("[GenericTrainer] calql package not found — fallback to CQL in d3rlpy.")\n            return _CQLConfig().create(device=device)\n        else:\n            # default safe baseline\n            return DQNConfig().create(device=device)\n\n    # ---- public API expected by online_loop ----\n    def add_transition(self, transition: dict):\n        \\"\\"\\"Add one transition; maybe trigger an update.\\"\\"\\"\n        try:\n            t = {\n                "observation": np.asarray(transition["state"], dtype=np.float32),\n                "action": int(transition["action"]),\n                "reward": float(transition["reward"]),\n                "next_observation": np.asarray(transition["next_state"], dtype=np.float32),\n                "terminal": bool(transition.get("done", transition.get("terminal", False))),\n            }\n        except Exception as e:\n            return {"updated": False, "error": f"bad transition: {e}"}\n\n        with self._buffer_lock:\n            self.online_buffer.append(t)\n            buf_len = len(self.online_buffer)\n            # first build as soon as we have enough for covering actions (done in perform_update)\n            need_update = (buf_len % self._update_frequency == 0) and (buf_len >= self._batch_size)\n\n        if need_update:\n            return self.perform_update()\n        return {"updated": False, "buffer_size": buf_len}\n\n    def _make_dataset(self, samples):\n        obs = np.asarray([s["observation"] for s in samples], dtype=np.float32)\n        next_obs = np.asarray([s["next_observation"] for s in samples], dtype=np.float32)\n        acts = np.asarray([int(s["action"]) for s in samples], dtype=np.int64).reshape(-1,1)\n        rews = np.asarray([float(s["reward"]) for s in samples], dtype=np.float32)\n        terms = np.asarray([bool(s["terminal"]) for s in samples], dtype=np.bool_)\n        if not terms.any():\n            # guarantee at least one episode boundary\n            terms = terms.copy()\n            # use timeouts channel for safety with newer d3rlpy\n            timeouts = np.zeros_like(terms, dtype=np.bool_)\n            timeouts[-1] = True\n        else:\n            timeouts = np.zeros_like(terms, dtype=np.bool_)\n        try:\n            return MDPDataset(observations=obs, actions=acts, rewards=rews,\n                              terminals=terms, timeouts=timeouts, next_observations=next_obs)\n        except TypeError:\n            try:\n                return MDPDataset(observations=obs, actions=acts, rewards=rews,\n                                  terminals=terms, timeouts=timeouts)\n            except TypeError:\n                return MDPDataset(observations=obs, actions=acts, rewards=rews,\n                                  terminals=(terms | timeouts))\n\n    def _first_build_indices(self):\n        \\"\\"\\"Pick a small warm-up batch that covers all seen action classes to set correct action_size.\\"\\"\\"\n        # collect first index for each unique action\n        first_idx = {}\n        for i, s in enumerate(self.online_buffer):\n            a = int(s["action"])\n            if a not in first_idx:\n                first_idx[a] = i\n        idxs = list(first_idx.values())\n        # fill to batch size with random uniques\n        if len(idxs) < min(self._batch_size, len(self.online_buffer)):\n            rest = [i for i in range(len(self.online_buffer)) if i not in idxs]\n            random.shuffle(rest)\n            need = min(self._batch_size, len(self.online_buffer)) - len(idxs)\n            idxs += rest[:need]\n        return idxs\n\n    def perform_update(self):\n        with self._buffer_lock:\n            buf_len = len(self.online_buffer)\n            if buf_len == 0:\n                return {"updated": False, "reason": "empty buffer"}\n\n            if not self._built:\n                # warm-up build with action coverage\n                warm_idxs = self._first_build_indices()\n                warm_batch = [self.online_buffer[i] for i in warm_idxs]\n                ds = self._make_dataset(warm_batch)\n                try:\n                    self.algo.build_with_dataset(ds)\n                except Exception:\n                    pass\n                self._built = True\n\n            # sample a normal batch\n            size = min(self._batch_size, buf_len)\n            batch_idxs = np.random.choice(buf_len, size=size, replace=False)\n            batch = [self.online_buffer[i] for i in batch_idxs]\n\n        # Create dataset & fit\n        ds = self._make_dataset(batch)\n        tr = None\n        try:\n            tr = self.algo.fit(ds, n_steps=self._update_steps,\n                               n_steps_per_epoch=self._update_steps,\n                               show_progress=False, save_interval=None)\n        except TypeError:\n            try:\n                tr = self.algo.fit(ds, n_steps=self._update_steps,\n                                   n_steps_per_epoch=self._update_steps,\n                                   show_progress=False)\n            except TypeError:\n                try:\n                    tr = self.algo.fit(ds, n_epochs=1, show_progress=False)\n                except TypeError:\n                    tr = self.algo.fit(ds, show_progress=False)\n\n        self._update_count += 1\n        avg_loss = 0.0\n        try:\n            if tr and hasattr(tr, "history"):\n                hist = getattr(tr, "history", {})\n                last = None\n                if isinstance(hist, dict):\n                    for k in ("loss","td_loss","imitator_loss"):\n                        if k in hist and hist[k]:\n                            last = float(hist[k][-1])\n                            break\n                if last is not None:\n                    self.training_losses.append(last)\n            if self.training_losses:\n                avg_loss = float(np.mean(list(self.training_losses)))\n        except Exception:\n            pass\n\n        return {"updated": True, "update_count": self._update_count,\n                "buffer_size": buf_len, "avg_loss": avg_loss}\n\n    def predict(self, state: np.ndarray) -> int:\n        try:\n            x = np.asarray(state, dtype=np.float32)\n            if x.ndim == 1:\n                x = x.reshape(1, -1)\n            a = self.algo.predict(x)\n            if hasattr(a, "__len__"):\n                return int(a[0])\n            return int(a)\n        except Exception:\n            return 0\n\n    def get_statistics(self):\n        return {\n            "algo": self._algo_name,\n            "updates_done": int(self._update_count),\n            "buffer_size": int(len(self.online_buffer)),\n            "avg_loss": float(np.mean(list(self.training_losses))) if self.training_losses else 0.0,\n        }\n'

def _write_patch_dir():
    tmpdir = tempfile.mkdtemp(prefix="online_patch_")
    with open(os.path.join(tmpdir, "sitecustomize.py"), "w", encoding="utf-8") as f:
        f.write(SITECUSTOMIZE)
    with open(os.path.join(tmpdir, "generic_trainer.py"), "w", encoding="utf-8") as f:
        f.write(GENERIC_TRAINER)
    return tmpdir

def _which_python():
    return sys.executable or "python"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", type=str, default="calql,dqn,double_dqn,nfq",
                    help="Comma-separated: calql,dqn,double_dqn,nfq")
    ap.add_argument("--mode", type=int, default=1)
    ap.add_argument("--duration", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--update-steps", type=int, default=20)
    ap.add_argument("--update-frequency", type=int, default=20)
    ap.add_argument("--device", type=str, default=None, help="cuda|cpu (default: auto)")
    ap.add_argument("--logdir", type=str, default="comparison_logs")
    args = ap.parse_args()

    algos = [x.strip().lower() for x in args.algos.split(",") if x.strip()]
    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    patch_dir = _write_patch_dir()
    print(f"[runner] patch dir: {{{{patch_dir}}}}")

    # Determine device
    dev = args.device
    if dev is None:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cuda"

    for algo in algos:
        print(f"\\n=== Running algo: {{{{algo}}}} ===")
        env = os.environ.copy()
        # Configure algo + online hyperparams for the generic trainer
        env["COMPARISON_ALGO"] = algo
        env["UPDATE_STEPS"] = str(args.update_steps)
        env["BATCH_SIZE"] = str(args.batch_size)
        env["UPDATE_FREQUENCY"] = str(args.update_frequency)
        env["D3RLPY_DEVICE"] = dev  # informative
        # Prepend our patch dir to PYTHONPATH so sitecustomize triggers
        env["PYTHONPATH"] = patch_dir + os.pathsep + env.get("PYTHONPATH", "")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logdir / f"{{{{algo}}}}_{{{{ts}}}}.stdout.log"
        with open(log_path, "w", encoding="utf-8") as logf:
            cmd = [_which_python(), "run_complete_evaluation.py", "--mode", str(args.mode), "--duration", str(args.duration)]
            print("[runner] CMD:", " ".join(cmd))
            print("[runner] logging to:", str(log_path))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
            for line in iter(proc.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                logf.buffer.write(line)
            ret = proc.wait()
            print(f"[runner] return code: {{{{ret}}}}")

    print(f"\\nAll done. Logs in: {{{{logdir.resolve()}}}}")
    print("If you want to re-run a single baseline: e.g., --algos dqn")

if __name__ == "__main__":
    main()
