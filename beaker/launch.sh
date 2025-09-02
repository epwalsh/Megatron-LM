#!/bin/bash

set -ex

script="${1:-beaker/llama3_8b_8k.sh}"

name=$(basename "$script")
# Remove file extension for naming.
name="${name%.*}"
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
    --workspace=ai2/google_benchmarks \
    --weka=oe-training-default:/weka/oe-training-default \
    --group=petew/B200_benchmarks \
    --group="petew/B200_benchmarks_${group_name}" \
    --priority=urgent \
    --task-timeout=120m \
    --env-secret='GOOGLE_CREDENTIALS=GOOGLE_CREDENTIALS' \
    --env-secret='BEAKER_TOKEN' \
    --beaker-image=petew/megatron-lm \
    --system-python \
    --install=beaker/install.sh \
    --replicas=5 \
    --leader-selection \
    --host-networking \
    --propagate-failure \
    --propagate-preemption \
    --synchronized-start-timeout='5m' \
    --gpu-type=b200 \
    --gpus=8 -- \
    "$script"

    # --replicas=2 \
    # --leader-selection \
    # --host-networking \
    # --propagate-failure \
    # --propagate-preemption \
    # --synchronized-start-timeout='5m' \
