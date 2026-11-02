"""
CLI Bridge to delegate tasks to Pi Coding Agent via Strata on LAN (172.20.20.120)
"""

import sys
import os
import argparse
import json
import time

sys.path.append(r"C:\Users\Korisnik\.gemini\mcp\pi_worker")
import pi_worker_mcp

def main():
    parser = argparse.ArgumentParser(description="Delegate task to Pi Worker")
    subparsers = parser.add_subparsers(dest="command")

    # delegate
    p_del = subparsers.add_parser("delegate")
    p_del.add_argument("--task-id", required=True)
    p_del.add_argument("--prompt", required=True)
    p_del.add_argument("--repo", default=r"D:\Kaggle")
    p_del.add_argument("--branch", default=None)
    p_del.add_argument("--provider", default="strata")
    p_del.add_argument("--model", default="qwen3.8-flash-next-coder-iq1_m")

    # status
    p_stat = subparsers.add_parser("status")
    p_stat.add_argument("--task-id", required=True)
    p_stat.add_argument("--tail", type=int, default=30)

    # wait
    p_wait = subparsers.add_parser("wait")
    p_wait.add_argument("--task-id", required=True)
    p_wait.add_argument("--timeout", type=int, default=180)

    # review
    p_rev = subparsers.add_parser("review")
    p_rev.add_argument("--task-id", required=True)

    args = parser.parse_args()

    if args.command == "delegate":
        res = pi_worker_mcp.handle_delegate_task({
            "task_id": args.task_id,
            "prompt": args.prompt,
            "repo_path": args.repo,
            "branch": args.branch,
            "provider": args.provider,
            "model": args.model
        })
        print(json.dumps(res, indent=2))

    elif args.command == "status":
        res = pi_worker_mcp.handle_get_task_status({
            "task_id": args.task_id,
            "tail_lines": args.tail
        })
        print(json.dumps(res, indent=2))

    elif args.command == "wait":
        start = time.time()
        print(f"Waiting for task {args.task_id} to finish (timeout: {args.timeout}s)...")
        while time.time() - start < args.timeout:
            res = pi_worker_mcp.handle_get_task_status({"task_id": args.task_id})
            status = res.get("status")
            if status in ["completed", "failed"] or "failed" in str(status):
                print(f"Task finished with status: {status}")
                print("\n=== Recent Logs ===")
                print(res.get("recent_logs", ""))
                return
            time.sleep(2)
        print(f"Timeout waiting for task {args.task_id}")

    elif args.command == "review":
        res = pi_worker_mcp.handle_review_result({"task_id": args.task_id})
        print(json.dumps(res, indent=2))

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
