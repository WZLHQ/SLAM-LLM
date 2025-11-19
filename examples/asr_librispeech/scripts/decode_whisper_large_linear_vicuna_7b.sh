#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/root/LLM-based-ASR/SLAM-LLM
cd $run_dir
code_dir=examples/asr_librispeech

speech_encoder_path=/root/autodl-tmp/pretrained_models/large-v3.pt
# llm_path=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08 # Llama-3.2-1B
llm_path=/root/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6 # Llama-3.2-1B-Instruct
special_experiment_key="A1"
output_dir=/root/autodl-tmp/outputs/Llama-3.2-1B-Instruct-librispeech-linear-steplrwarmupkeep1e-4-whisper-largev3-$special_experiment_key

ckpt_path=$output_dir/asr_epoch_2_step_4244
split=librispeech_test-clean
val_data_path=/root/autodl-tmp/jsonl_data/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam4

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
        ++train_config.val_batch_size=4 \
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


export HF_ENDPOINT=https://hf-mirror.com
python $code_dir/compute_asr_metrics.py \
        --wer_or_cer wer \
        --predictions_file_path ${decode_log}_pred \
        --references_file_path ${decode_log}_gt \
        --results_file_path $ckpt_path/RESULTS.txt \

