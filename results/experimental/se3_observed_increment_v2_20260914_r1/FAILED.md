# Failed execution receipt

This run stopped before any test metric was written or inspected.

Failure: the legacy matched-donor helper grouped all independent records under
the synthetic `window_id=-1`, then its one-position scene-sorted rotation
selected a same-scene donor.  The fail-closed control check raised
`RuntimeError: matched shuffle selected a same-scene donor`.

The partial directory is retained and is not a scientific result.  The fixed
runner uses an exact cross-scene derangement selected by action-magnitude
matching before scoring; its output is written to a new revision.
