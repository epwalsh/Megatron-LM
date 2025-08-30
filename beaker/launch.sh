#!/bin/bash

set -ex

script="${1:-mixtral/8x7b_8k.sh}"
# Valid script names are:
# - llama3/8b_8k.sh
# - llama3/70b_8k.sh
# - llama3/8b_128k.sh
# - mixtral/8x7b_8k.sh

# Remove file extension for naming.
name="${script%.*}"
# Replace slashes in path with dashes.
name="${name//\//-}"
# Replace underscores with dashes for run name.
name="${name//_/-}"
# Keep group name with underscores.
group_name="${name//-/_}"

gantry run \
    --show-logs \
    --yes \
    --allow-dirty \
    --name="${name}-$(date +%Y%m%d-%H%M%S)" \
    --description="Megatron-LM ${name}" \
    --group=petew/B200_benchmarks \
    --group="petew/B200_benchmarks_${group_name}" \
    --priority=urgent \
    --task-timeout=120m \
    --env-secret='GOOGLE_CREDENTIALS=GOOGLE_CREDENTIALS' \
    --env-secret='BEAKER_TOKEN' \
    --beaker-image=petew/megatron-lm \
    --system-python \
    --install=beaker/install.sh \
    --gpu-type=b200 \
    --gpus=8 -- \
    echo "Hello, World!"


    # --replicas=2 \
    # --leader-selection \
    # --host-networking \
    # --propagate-failure \
    # --propagate-preemption \
    # --synchronized-start-timeout='5m' \
