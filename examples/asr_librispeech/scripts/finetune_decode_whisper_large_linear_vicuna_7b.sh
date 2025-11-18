#!/bin/bash



special_experiment_key=$1
stage=$2 # 1 for training, 2 for decoding, and 3 for evaluation
stop_stage=$3
speech_encoder=$4 # select from hubert_xtralarge_ll60k_finetune_ls960 / whisper_large_v3 / wavlm_large / ...
LLM=$5 # select from Llama-3.2-1B / Llama-3.2-1B-Instruct / Llama-3.2-3B / vicuna-7b-v1.5 / ...
train_jsonl=$6 # select from librispeech_train-clean-100 / librispeech_train-clean-360 / librispeech_train-other-500 (TODO)
valid_jsonl=$7 # select from librispeech_dev-clean / librispeech_dev-other / librispeech_dev-all (TODO)
test_jsonl=$8 # select from librispeech_test-clean / librispeech_test-other / librispeech_test-all (TODO)
train_val_batch_size=$9 # e.g., 6
eval_model_dir=${10} # e.g., asr_epoch_2_step_4244
num_epochs=${11} # e.g., 5


if [[ "$speech_encoder" == *"whisper"* ]]; then
    export PYTHONPATH=/root/LLM-based-ASR/whisper:$PYTHONPATH
    encoder_name="whisper"
elif [[ "$speech_encoder" == *"hubert"* ]]; then
    export PYTHONPATH=/root/LLM-based-ASR/fairseq:$PYTHONPATH
    encoder_name="hubert"
elif [[ "$speech_encoder" == *"wavlm"* ]]; then
    export PYTHONPATH=/root/LLM-based-ASR/fairseq:$PYTHONPATH
    encoder_name="wavlm"
else
    echo "Unsupported speech encoder: $speech_encoder"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
export OMP_NUM_THREADS=1

# debug setting for multiple gpus
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

run_dir=/root/LLM-based-ASR/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech
speech_encoder_path=/root/autodl-tmp/pretrained_models/speech/${speech_encoder}.pt
llm_path=/root/autodl-tmp/pretrained_models/LLM/${LLM}
train_data_path=/root/autodl-tmp/jsonl_data/${train_jsonl}.jsonl
val_data_path=/root/autodl-tmp/jsonl_data/${valid_jsonl}.jsonl
output_dir=/root/autodl-tmp/outputs/${speech_encoder}-${LLM}-$special_experiment_key

ckpt_path=${output_dir}/$eval_model_dir
val_data_path=/root/autodl-tmp/jsonl_data/${test_jsonl}.jsonl
decode_log=$ckpt_path/decode_${test_jsonl}_beam4

# TODO: are you sure about this? 
if [[ "$LLM"==*"1B"* ]] || [[ "$LLM"==*"1b"* ]]; then
    llm_dim=2048
elif [[ "$LLM"==*"3B"* ]] || [[ "$LLM"==*"3b"* ]]; then
    llm_dim=3072
elif [[ "$LLM"==*"7B"* ]] || [[ "$LLM"==*"7b"* ]]; then
    llm_dim=4096
else
    echo "Unsupported LLM dimension for $LLM"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then

    echo "Starting fine-tuning with $speech_encoder and $LLM ..."
    
    hydra_args="
    hydra.run.dir=$output_dir \
    ++model_config.llm_name=$LLM \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.encoder_projector_ds_rate=5 \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_dim=1280 \
    ++model_config.encoder_projector=linear \
    ++dataset_config.dataset=speech_dataset \
    ++dataset_config.train_data_path=$train_data_path \
    ++dataset_config.val_data_path=$val_data_path \
    ++dataset_config.input_type=mel \
    ++dataset_config.mel_size=128 \
    ++train_config.model_name=asr \
    ++train_config.num_epochs=$num_epochs \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.batching_strategy=custom \
    ++train_config.warmup_steps=1000 \
    ++train_config.total_steps=100000 \
    ++train_config.lr=1e-4 \
    ++train_config.validation_interval=1000 \
    ++train_config.batch_size_training=$train_val_batch_size \
    ++train_config.val_batch_size=$train_val_batch_size \
    ++train_config.num_workers_dataloader=2 \
    ++train_config.output_dir=$output_dir \
    ++metric=acc \
    ++train_config.use_fp16=true \
    "

    # -m debugpy --listen 5678 --wait-for-client
    if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then

        # 单GPU训练（使用调试器）
        # python -m debugpy --listen 5678 --wait-for-client $code_dir/finetune_asr.py \
        #     --config-path "conf" \
        #     --config-name "prompt.yaml" \
        #     $hydra_args

        # 单GPU训练（不使用调试器）
        python $code_dir/finetune_asr.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            $hydra_args
    else
        # 多GPU训练 ?
        torchrun \
            --nnodes 1 \
            --nproc_per_node 2 \
            --master_port=29503 \
            $code_dir/finetune_asr.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            ++train_config.enable_fsdp=false \
            ++train_config.enable_ddp=true \
            ++train_config.use_fp16=true \
            $hydra_args
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

    echo "Decoding with the fine-tuned model..."

    # -m debugpy --listen 5678 --wait-for-client
    python $code_dir/inference_asr_batch.py \
            --config-path "conf" \
            --config-name "prompt.yaml" \
            hydra.run.dir=$ckpt_path \
            ++model_config.llm_name="Llama-3.2-1B-Instruct" \
            ++model_config.llm_path=$llm_path \
            ++model_config.llm_dim=2048 \
            ++model_config.encoder_name=whisper \
            ++model_config.encoder_projector_ds_rate=5 \
            ++model_config.encoder_path=$speech_encoder_path \
            ++model_config.encoder_dim=1280 \
            ++model_config.encoder_projector=linear \
            ++dataset_config.dataset=speech_dataset \
            ++dataset_config.val_data_path=$val_data_path \
            ++dataset_config.input_type=mel \
            ++dataset_config.mel_size=128 \
            ++dataset_config.inference_mode=true \
            ++train_config.model_name=asr \
            ++train_config.freeze_encoder=true \
            ++train_config.freeze_llm=true \
            ++train_config.batching_strategy=custom \
            ++train_config.num_epochs=1 \
            ++train_config.val_batch_size=$train_val_batch_size \
            ++train_config.num_workers_dataloader=2 \
            ++train_config.output_dir=$output_dir \
            ++decode_log=$decode_log \
            ++ckpt_path=$ckpt_path/model.pt \
            # ++peft_ckpt=$ckpt_path \
            # ++train_config.use_peft=true \
            # ++train_config.peft_config.r=32 \
            # ++dataset_config.normalize=true \
            # ++model_config.encoder_projector=q-former \
            # ++dataset_config.fix_length_audio=64 \
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then

    echo "Evaluating the decoding results..."
    export HF_ENDPOINT=https://hf-mirror.com
    python $code_dir/compute_asr_metrics.py \
            --predictions_file_path ${decode_log}_pred \
            --references_file_path ${decode_log}_gt \
            --results_file_path $ckpt_path/RESULTS.txt \

fi
