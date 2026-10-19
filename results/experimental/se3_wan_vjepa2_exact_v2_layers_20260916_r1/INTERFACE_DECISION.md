# C1/C2 interface decision

The existing C1 `ActionDynamicsProbe` is not a closed recurrent state update.
Its input is a streaming-context token sequence, while its output is a pooled
isolated-target feature. Equal channel dimensions do not make those roles
interchangeable, so the C1 output is not fed back as a new context.

The C2 `PathIntegrationProbe` already contains a repeated GRU hidden-state
transition, but its public API previously exposed only an entire rollout. This
experiment makes the existing initializer, one-step updater, and decoder
explicit as `initialize_state`, `update_state`, and `decode_state`; `forward`
uses those same methods. The diagnostic checkpoint is trained with only the
first real one-step target B. No B/C, composition, order, cycle, or rollout loss
is used. The same checkpoint and updater are then used for C1 calibration,
sequential/direct composition, and swapped-order evaluation.

Original C1 code and checkpoints remain unchanged. The new head is therefore a
one-step-supervised C2-structured diagnostic head, not a relabelled C1 model and
not a multi-step-supervised C2 result.
