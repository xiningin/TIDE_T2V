# Wan2.2-TI2V-5B arm status: blocked

Wan2.2-TI2V-5B weights are available, but the official Wan2.2 inference code
checkout is missing. The existing Wan2.1 code does not support `ti2v-5B`, so the
`wan22_raw` and `wan22_ours` arms cannot be launched reliably. Do not force these
weights through the Wan2.1 adapter.
