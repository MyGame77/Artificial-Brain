# Android backend

The Android phone runs the model. GitHub Pages is only the frontend.

Requirements: Termux, an Android-built llama.cpp `llama-server`, the local GGUF model, and a Cloudflare Tunnel protected by an access policy.

1. Keep the GGUF model on the phone.
2. Check the model path in `start-model.sh`.
3. Run `chmod +x start-model.sh && ./start-model.sh` in Termux.
4. The model listens only on `127.0.0.1:8080`.

Do not expose port 8080 directly to the internet or port-forward it. Put authenticated HTTPS access in front of it. Set the Pages frontend API URL to that final HTTPS endpoint only after tunnel protection is configured.

The phone must remain powered, online, and allowed to run Termux in the background.