"""Active Spatial frozen-representation probing.

This package deliberately does not reuse InsScene feature caches or labels.
It is a small, standalone pipeline for collecting one fixed Active Spatial
state bank, extracting frozen checkpoint activations, and fitting probes.
"""

PROTOCOL_VERSION = "active_repr_probe_mvp_v1"
