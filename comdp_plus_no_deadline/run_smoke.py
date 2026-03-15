import subprocess
import sys

from comdp_plus_no_deadline.scenarios import PRESETS


def run_smoke(runs=1, max_steps=40):
    print("=== No-Deadline CoMDP+ smoke run ===")
    for name, config in PRESETS.items():
        domain = config["domain"]
        object_amount = config["object_amount"]

        print(f"[{name}]")
        for domain_type in ("regular", "combination"):
            cmd = [
                sys.executable,
                "-m",
                "comdp_plus_no_deadline.run_no_deadline",
                "--domain",
                domain,
                "--object_amount",
                str(object_amount),
                "--domain_type",
                domain_type,
                "--runs",
                str(runs),
                "--max_steps",
                str(max_steps),
                "--seed",
                "123",
            ]
            completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
            print(f"  {domain_type} exit={completed.returncode}")
            print(completed.stdout.strip())


if __name__ == "__main__":
    run_smoke()

