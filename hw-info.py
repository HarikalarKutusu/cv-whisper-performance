###########################################################################
# gpu-info.py
#
# Tooling to get the GPU info
#
# This script is part of Common Voice ToolBox Package
#
# github: https://github.com/HarikalarKutusu/cv-whisper-performance
# Copyright: (c) Bülent Özden, License: AGPL v3.0
###########################################################################

import psutil
import torch
from lib import bytes2gb

def main() -> None:
    """Main process which shows you GPU's and their capabilities."""
    print("\n=== CPU ===")
    print(f"Real cores\t: {psutil.cpu_count(logical=False)}")
    print(f"Logical cores\t: {psutil.cpu_count(logical=True)}")
    print(f"Freq\t\t: {psutil.cpu_freq(percpu=False).current}")
    print(f"Load avg.\t: {psutil.getloadavg()}")

    print("\n=== VIRTUAL MEMORY ===")
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"Total\t\t: {bytes2gb(vm.total)} GB")
    print(f"Used\t\t: {bytes2gb(vm.used)} GB")
    print(f"Free\t\t: {bytes2gb(vm.free)} GB")
    print(f"Swap total\t: {bytes2gb(swap.total)} GB")
    print(f"Swap used\t: {bytes2gb(swap.used)} GB")

    print("\n=== GPUs ===")
    gpu_cnt: int = torch.cuda.device_count()
    cur_gpu: int = torch.cuda.current_device()
    print(f"GPUs\t\t: {torch.cuda.device_count()}")
    print(f"Current\t\t: {cur_gpu} - {torch.cuda.get_device_name(cur_gpu)}")
    for gpu_no in range(gpu_cnt):
        gpu_mem = torch.cuda.mem_get_info(gpu_no)
        gpu_comp = torch.cuda.get_device_capability(gpu_no)
        print(f"\n=== GPU #{gpu_no} - {torch.cuda.get_device_name(gpu_no)}")
        print(f"VRAM Free\t: {bytes2gb(gpu_mem[0])} GB")
        print(f"VRAM Total\t: {bytes2gb(gpu_mem[1])} GB")
        print(f"Compute Cap.\t: {gpu_comp[0]}.{gpu_comp[1]}")


# Entry point
if __name__ == "__main__":
    main()
