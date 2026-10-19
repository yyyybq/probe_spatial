# Reproduce the frozen paper package

```bash
MPLCONFIGDIR=/tmp/mpl-se3-paper-final \
/mnt/umm/users/yinbaiqiao/.conda/envs/probe-spatial/bin/python \
  -m analyses.se3_protocol_v2.wan_vjepa2_paper_final
```

This command only reads frozen artifacts and regenerates audits, tables, figures, text, and hashes. It performs no training, model inference, or feature extraction.
