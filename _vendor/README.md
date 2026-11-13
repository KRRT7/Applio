# Vendored libraries

Applio carries a copy of some third-party code here, imported through the
`_vendor` namespace (the layout pip uses in `src/pip/_vendor`). Unlike pip,
this directory is **not** managed by a sync tool: the code is maintained in
place.

| Library | Packages | Origin |
|---|---|---|
| deep-live-cam 2.1.5 | `deep_live_cam`, `modules` | Deep-Live-Cam, branch `headless-library` at commit `047c551` (a headless fork of hacksider/Deep-Live-Cam with HyperSwap support). That checkout was deleted and never pushed, so this is the only copy. |

## Rules

- **Imports use the `_vendor` namespace**, both from Applio
  (`from _vendor import deep_live_cam`) and inside the vendored code
  (`from _vendor.modules.x import y`). Keep it that way when editing, so the
  code never depends on a top-level `modules` package.
- **Edit in place, deliberately.** There is no upstream to fix things in, so
  changes go straight into these files. Keep them focused and say in the
  commit message that vendored code changed.
- **Dependencies are not vendored.** Deep-Live-Cam's runtime dependencies are
  declared in Applio's `pyproject.toml` (the `faceswap` dependency group).
- **License:** `deep_live_cam/LICENSE` covers both packages. Deep-Live-Cam is
  **AGPL-3.0**; Applio itself is MIT. Distributing Applio with this code, or
  serving it to others over a network, brings the AGPL terms into play for
  the combined work.

## Runtime configuration

Deep-Live-Cam downloads its models to `DLC_MODELS_DIR`; the Face Swap tab
sets this to `rvc/models/faceswap` before importing it.
