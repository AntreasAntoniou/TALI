import os
import subprocess
import fire
from concurrent.futures import ProcessPoolExecutor


def run_command(set_name, start_idx, end_idx, num_workers=256):
    cmd = f"/opt/conda/envs/minimal-ml-template/bin/python tali_wit/generate_hf_cache_dataset.py --set_name {set_name} --start_idx {start_idx} --end_idx {end_idx} --num_workers {num_workers}"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def generate_args(N, set_name_prefix, start_idx_base, end_idx_base, step):
    set_name_list = [f"{set_name_prefix}" for i in range(N)]
    start_idx_list = [start_idx_base + i * step for i in range(N)]
    end_idx_list = [end_idx_base + i * step for i in range(N)]

    return set_name_list, start_idx_list, end_idx_list


def main(N, set_name_prefix, start_idx_base, end_idx_base, step, num_workers=256):
    set_name_list, start_idx_list, end_idx_list = generate_args(N, set_name_prefix, start_idx_base, end_idx_base, step)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in executor.map(run_command, set_name_list, start_idx_list, end_idx_list, [16]*N):
            print(result)


if __name__ == "__main__":
    fire.Fire(main)
