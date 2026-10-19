# Feature lineage audit

+Status: `COMPLETE_WITH_CAVEAT`

+An exact raw-ID feature request manifest has been frozen.  No historical feature is accepted by filename/window/tail similarity.  Reuse requires a sidecar or indexed extractor record proving each requested actual raw ID.

+- Requests: 121748
+- Unique requested `(model, layer, timestep, scene, raw-ID)` tuples: 108840
+- Missing/reuse resolution: `NOT_RUN`
+- Fallback policy: forbidden
+- SCO submission prerequisite: `BLOCKED_ENVIRONMENT` because `/mnt/umm/users/yinbaiqiao/submit.sh` is 1 bytes.  Only a verified Cursor History copy may restore it.

+The next valid action is an exact sidecar audit followed by a DINO-only raw-ID extraction/load smoke for missing requests.
+