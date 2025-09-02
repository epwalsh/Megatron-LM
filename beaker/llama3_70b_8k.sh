#!/bin/bash

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export OMP_NUM_THREADS=8
#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}
export NCCL_NVLS_ENABLE=0

TOKENIZER_ARG=${3:-"MOCK"} # Path to tokenizer model, or "MOCK"
DATA_ARG=${4:-"MOCK"}     # Data prefix, or "MOCK"

# Distributed training setup
if [[ -n "$BEAKER_REPLICA_COUNT" ]]; then
    GPUS_PER_NODE=8
    NUM_NODES="$BEAKER_REPLICA_COUNT"
    NODE_RANK="$BEAKER_REPLICA_RANK"
    MASTER_ADDR="$BEAKER_LEADER_REPLICA_HOSTNAME"
    MASTER_PORT=29400
else
    GPUS_PER_NODE=8
    NUM_NODES=1
    NODE_RANK=${NODE_RANK:-0}
    MASTER_ADDR=${MASTER_ADDR:-localhost}
    MASTER_PORT=${MASTER_PORT:-6000}
fi
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="beaker/train.py"

# Fixed model and training parameters
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=$(($WORLD_SIZE*$MICRO_BATCH_SIZE))
DTYPE="bf16"
SEQ_LENGTH=8192

# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="/weka/oe-training-default/petew/google_benchmarks/benchmark_cache_llama3_8b_fp8"
mkdir -p "$DATA_CACHE_PATH"

TORCHRUN_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --num-layers 80
    --hidden-size 8192
    --ffn-hidden-size 28672
    --num-attention-heads 64
    --group-query-attention
    --num-query-groups 8
    --kv-channels 128
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $SEQ_LENGTH
    --position-embedding-type rope
    --rotary-base 1000000
    --rotary-percent 1.0
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --swiglu
    --init-method-std 0.0134
    --attention-backend fused
    --apply-layernorm-1p 
    --untie-embeddings-and-output-weights
    --disable-bias-linear
)

DISTRIBUTED_ARGS=(
    # --init-model-with-meta-device
    # Data parallelism with torch FSDP2.
    # --data-parallel-sharding-strategy optim_grads_params
    # --use-torch-fsdp2
    # --no-gradient-accumulation-fusion
    # Data parallelism with megatron FSDP.
    # --use-megatron-fsdp
    # --init-model-with-meta-device
    # --use-distributed-optimizer
    # --overlap-grad-reduce
    # --overlap-param-gather
    # Data parallelism, ZeRO-1 style.
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
    # Tensor parallelism.
	--tensor-model-parallel-size 2
	--pipeline-model-parallel-size $((NUM_NODES*4))
    # Context parallelism.
    --context-parallel-size 1
)

ACTIVATION_CHECKPOINTING_ARGS=(
    # --recompute-granularity full
    # --recompute-method block
    # --recompute-num-layers 80
    # --recompute-granularity selective
    # --recompute-modules layernorm mlp
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters 100
    --lr-decay-iters 1000
    --lr-warmup-iters 20
    --lr 0.00015
    --min-lr 0.00001
    --decoupled-lr 5.0e-4      # Specific to decoupled AdamW, ensure optimizer is compatible
    --decoupled-min-lr 4.5e-5  # Specific to decoupled AdamW
    --lr-decay-style cosine
    --clip-grad 1.0
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --cross-entropy-loss-fusion
    --calculate-per-token-loss 
    --manual-gc 
    --empty-unused-memory-level 1 
    --exit-duration-in-mins 235
    --enable-cuda-graph
    --no-check-for-nan-in-loss-and-grad
)

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
        "--bf16"
        "--grad-reduce-in-bf16"
    )
elif [[ "$DTYPE" == "bf16" ]]; then
    DTYPE_ARGS+=(
        "--bf16"
        "--grad-reduce-in-bf16"
    )
else
    echo "unsuported DTYPE: $DTYPE"
    exit 1
fi


# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 128256" 
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 128256"
    )
fi

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --beaker-log-interval 5
    --eval-iters 0
    --eval-interval 1000
    --save-interval 1000
    --log-throughput
    --profile
    --profile-step-start 4
    --profile-step-end 6
    --ckpt-format torch_dist 
    # --ckpt-format fsdp_dtensor
    --distributed-timeout-minutes 60
)

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: script $PRETRAIN_SCRIPT_PATH not found"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and that $PRETRAIN_SCRIPT_PATH is present."
    exit 1
fi

# Run the training command
torchrun ${TORCHRUN_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${ACTIVATION_CHECKPOINTING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
