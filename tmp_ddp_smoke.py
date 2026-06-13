"""Minimal 4-GPU NCCL smoke test (no Lightning, no dataset)."""
import os
import time
import torch
import torch.distributed as dist


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    print(f"[rank{rank}] init_process_group...", flush=True)
    dist.init_process_group(backend="nccl", timeout=__import__("datetime").timedelta(seconds=120))
    print(f"[rank{rank}] init done. world={world}", flush=True)

    # Broadcast (analog of DDP param broadcast at SeqNum=1)
    t = torch.full((1,), float(rank), device=f"cuda:{local_rank}")
    t0 = time.time()
    print(f"[rank{rank}] broadcasting...", flush=True)
    dist.broadcast(t, src=0)
    print(f"[rank{rank}] broadcast done in {time.time()-t0:.2f}s, value={t.item()}", flush=True)

    # Allreduce
    t = torch.full((1,), float(rank + 1), device=f"cuda:{local_rank}")
    dist.all_reduce(t)
    print(f"[rank{rank}] allreduce done, sum={t.item()}", flush=True)

    dist.barrier()
    print(f"[rank{rank}] barrier done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
