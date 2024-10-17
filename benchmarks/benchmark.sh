#!/bin/bash

USER_HOME=~/lexu
NOW=$( date '+%F-%H:%M:%S' )
LOG=log/$NOW/
mkdir -p $LOG
MODEL="/root/models/llama-2-7b-hf"

# wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -O /tmp/ShareGPT_V3_unfiltered_cleaned_split.json

# pip install -U "huggingface_hub[cli]"
# huggingface-cli login


for RANGE in 754:754 #64:128 128:256 256:512
# for RANGE in 2048:2048
do
    # unset VLLM_USE_VINEYARD_CACHE
    # python3 benchmark_prefix_caching.py --model ${MODEL} --num-prompts 1  --repeat-count 100 --input-length-range ${RANGE} --max-num-seqs 1 > ${LOG}/default-R${RANGE}.log 2>&1
    
    # unset VLLM_USE_VINEYARD_CACHE
    # python3 benchmark_prefix_caching.py --model ${MODEL} --num-prompts 1 --repeat-count 100 --enable-prefix-caching --input-length-range ${RANGE} --max-num-seqs 1 > ${LOG}/default-prefix-R${RANGE}.log 2>&1

    # #default chunked prefill
    # unset VLLM_USE_VINEYARD_CACHE
    # python3 benchmark_prefix_caching.py --model ${MODEL} --num-prompts 1 --repeat-count 100 --enable-chunked-prefill --input-length-range ${RANGE} --max-num-seqs 1 > ${LOG}/default-chunked-R${RANGE}.log 2>&1


    # #v1 shared memory
    source $USER_HOME/v1_config.sh
    # python benchmark_prefix_caching.py --model meta-llama/llama-2-7b-hf  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1 --repeat-count 100 --enable-chunked-prefill --input-length-range ${RANGE} --max-num-seqs 1 > ${LOG}/v1-R${RANGE}.log 2>&1
    # gdb --args python benchmark_prefix_caching.py --model meta-llama/llama-2-7b-hf  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1 --repeat-count 100 --enable-chunked-prefill --input-length-range ${RANGE} --max-num-seqs 1
    python3 benchmark_prefix_caching.py --model ${MODEL} --num-prompts 1 --repeat-count 100 --enable-chunked-prefill --input-length-range ${RANGE} --max-num-seqs 1 > ${LOG}/v1-R${RANGE}.log 2>&1

    # # ##v2 fs
    # source $USER_HOME/v2_config.sh
    # # python benchmark_prefix_caching.py --model meta-llama/llama-2-7b-hf  --dataset-path /tmp/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1 --repeat-count 100 --enable-chunked-prefill --input-length-range ${RANGE} --max-num-seqs 1 > ${LOG}/v2-R${RANGE}.log 2>&1
    # python3 benchmark_prefix_caching.py --model ${MODEL} --num-prompts 1 --repeat-count 100 --enable-chunked-prefill --input-length-range ${RANGE} --max-num-seqs 1 > ${LOG}/v2-R${RANGE}.log 2>&1

done
