# VISTA arm status: blocked

The `VistaAgent` in this repository is part of the TIDE training loop. It does
not yet expose a standalone VISTA adapter that takes a benchmark prompt plus a
fixed paired seed and returns a single final video. As a result, the
`sana_vista` arm cannot currently be launched under matched seeds; that adapter
needs to be implemented and frozen first.
