# Strict B2 Real Smoke

Status: **NOT RUN — scheduler create API returned HTTP 403 before allocating a worker.**

The authorized shared project identity was accepted as policy-compliant, but neither bounded one-GPU create attempt returned a job ID. Therefore no compute-node CUDA or exact-object AOSS gate could run, and the strict-B2 smoke prerequisite remained false.

No strict manifest or smoke sample artifact was generated in this tag. This report does not claim any real-data result for:

- `n_valid > 0`;
- target object raw pixel count equal to zero;
- target-mask mutation invariance;
- past-visible-mask sensitivity;
- exclusion of target-frame depth/object pixels from the centroid;
- VFM-independent GT;
- fixed target across p8/p12/p16/p24;
- streaming-prefix or target-isolated feature loading;
- real C/B2 DataLoader success.

The restored implementation and its unit tests pass, but synthetic/unit validation is not presented as a real strict-B2 experiment. No training was started.
