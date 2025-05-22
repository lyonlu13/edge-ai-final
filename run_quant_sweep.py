import os
import subprocess
from datetime import datetime
import argparse

# This script runs a series of quantization tests with different nbits and group sizes.
# It captures the output and saves it to log files with timestamps.
# Usage: python run_quant_sweep.py --device cuda:0
parser = argparse.ArgumentParser(description="Batch run quantization tests.")
parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device to use (default: cuda:0)")
args = parser.parse_args()

log_dir = "quant_log"
os.makedirs(log_dir, exist_ok=True)

nbits_list = [4, 5, 6, 8]
group_sizes = [32, 64, 128, 256, 512, 1024, 2048]

for nbits in nbits_list:
    for group_size in group_sizes:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{nbits}_{group_size}_{timestamp}.log"
        log_path = os.path.join(log_dir, log_filename)

        print(f"Running quantization with nbits={nbits}, group_size={group_size}, device={args.device}...")

        result = subprocess.run(
            ["python", "enc_result.py", "--nbits", str(nbits), "--group_size", str(group_size), "--device", args.device],
            capture_output=True,
            text=True,
            shell=True
        )

        with open(log_path, "w") as f:
            f.write(f"=== nbits={nbits}, group_size={group_size}, device={args.device}, timestamp={timestamp} ===\n\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
