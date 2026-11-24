# HU_D04 toe-vertical candidate: one bounded Flux smoke test

This repository contains the minimum project source/assets and candidate motion for an explicitly authorized **single GPU / 64 environments / 20 PPO updates / 1200-second worker ceiling** trial. No automatic retries, pretrained policy, real-robot deployment or quality claim.

## Input

- W&B artifact: `zhul136-university-of-science-and-technology-of-china/robot-motion-edits/gui00528900_toe_vertical_parallel_candidate:v0`.
- Motion SHA-256: `68e6d580354a86caeed1b94e7869475fb3220c1fa4dc95894057301aed62fb35`.
- 187 frames, 50 FPS, 29 joints; see motion validation JSON for CPU kinematic checks and limitations.
- Source/assets come from the user's `upup1124` transfer. Existing robot/mesh author notices and project metadata are retained; this publication does not relicense those assets.

## Runtime

Flux official image: IsaacSim 5.1 / IsaacLab 2.3.2 / Python 3.11 / Torch 2.7.0. The vendored RSL-RL 3.1.2 is the version requested by upstream IsaacLab v2.3.2; its BSD-3-Clause license is included. Wheels retain their bundled third-party notices. Only distributions absent from the image are installed offline; image Torch/CUDA/NumPy remain unchanged. Dependency constraints and CUDA are checked on the worker before simulator startup.

`flux/input_manifest.json` verifies every published training input and runtime file. The source provenance manifest hashes the original transfer files; the input manifest reflects the small bounded-run changes.

## Entry

`gm-run limedit/flux/run_smoke.py --execute --commit PINNED_COMMIT --upload-url SCOPED_SINGLE_FILE_PUT`

The actual short-lived signed output URL is supplied privately in task configuration, never committed. No GitHub/W&B/Flux API key is sent to training code. Runtime output is one ZIP <=100 MiB with the final/last checkpoint, YAML configs, TensorBoard metrics, update counter and completion/status records. Raw simulator logs are not published.

The watchdog includes dependency setup and simulator initialization but begins after platform code checkout. Platform startup/checkout/release billing is separate. The worker is terminated at its bound and cannot self-retry. A completed smoke test only proves the training chain; stable toe contact, tracking, balance and safe robot deployment still require separate validation and training authorization.

Headless-only changes disable visualization markers and a remotely fetched ground visual material. Physics, rewards, randomization, policy sizes and rollout length are unchanged. The wrapper counts successful PPO updates, records finite losses and saves an explicit final checkpoint after exactly twenty updates.