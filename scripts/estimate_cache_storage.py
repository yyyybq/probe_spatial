#!/usr/bin/env python3
"""Estimate dense feature-cache capacity before launching extraction."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--channels", type=int, required=True)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--copies", type=float, default=1.0,
                        help="Equivalent cache copies, e.g. normal+shuffled=2")
    parser.add_argument("--bytes-per-value", type=int, default=2)
    args = parser.parse_args()
    total = (args.scenes * args.frames * args.height * args.width * args.channels
             * args.layers * args.copies * args.bytes_per_value)
    print(f"bytes={int(total)} GiB={total / 2**30:.2f} TiB={total / 2**40:.3f}")


if __name__ == "__main__":
    main()
