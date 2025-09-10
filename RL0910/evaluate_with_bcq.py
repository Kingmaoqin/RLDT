# -*- coding: utf-8 -*-
"""
evaluate_with_bcq.py
Date: 2025-08-14

CLI wrapper that applies BCQ policy patch, then calls evaluation.run_evaluation()
so you can keep your original evaluation pipeline and reports.
"""

import argparse
import evaluation as E
from evaluation_bcq_minipatch import apply as apply_bcq_patch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_dir', type=str, default='./output/models')
    ap.add_argument('--output_dir', type=str, default='./evaluation_results')
    ap.add_argument('--train_models', action='store_true', help='Optionally run training inside evaluation (kept for parity).')
    ap.add_argument('--n_episodes', type=int, default=100, help='Policy evaluation episodes (evaluation.py will read this through its config).')
    args = ap.parse_args()

    # Apply BCQ patch (if policy exists, it will be used for policy evaluation)
    patched = apply_bcq_patch(E, args.model_dir)
    if not patched:
        print("[BCQ-PATCH] Proceeding without BCQ; evaluation will use original q_network decisions.")

    # Delegate to your existing evaluation entrypoint
    E.run_evaluation(model_dir=args.model_dir, output_dir=args.output_dir, train_models=args.train_models)

if __name__ == "__main__":
    main()
