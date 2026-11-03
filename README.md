# Artificial Brain

A from-scratch neural-network and language-model research project, with an Android-hosted local model backend and a GitHub Pages web interface.

## Architecture
- GitHub Pages serves the static web interface.
- The Android phone runs llama.cpp and the local GGUF model.
- Cloudflare Tunnel with access protection should provide the HTTPS backend endpoint.
- Python neural-network and transformer source is included.

GitHub Pages cannot run Python or host model weights. The model binary stays on the phone. The Pages interface is published with a placeholder backend URL until the secure phone endpoint is configured.

See [Android setup](android/README.md).

## Excluded
Model binaries, checkpoints, chat history, personal memory files, secrets, caches, and runtime logs are not published.