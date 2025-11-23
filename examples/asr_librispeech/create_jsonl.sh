# 当前目录下有多个文件夹，每个文件夹的目录结构如下所示：
# ├── speaker_id
# │   ├── chapter_id
# │   │   ├── speaker_id-chapter_id-utterance_id1.flac
# │   │   ├── speaker_id-chapter_id-utterance_id2.flac
# │   │   ├── ...
# │   │   ├── speaker_id-chapter_id.txt # 包含每个utterance的转录文本，格式如下：
# │   │   # speaker_id-chapter_id-utterance_id1 transcription text
# │   │   # speaker_id-chapter_id-utterance_id2 transcription text
# │   │   # ...

# 需要将这些数据转换为jsonl格式（名称为librispeech—文件夹名称），每行包含以下字段：
# {
#   "key": "speaker_id-chapter_id-utterance_id1", # 音频文件的名称（不含扩展名）
#   "source": ".../speaker_id/chapter_id/speaker_id-chapter_id-utterance_id1.flac", # 音频文件的绝对路径
#   "text": "transcription text"
# }

# tODO implement multiprocess

#!/bin/bash

output_dir="/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/100_global_shared_file/140_SLAM_ASR_datasets/141_librispeech/jsonl_data"
speech_file='/media/rosie/d921a251-72e5-45d8-9e41-0309cf76c6b4/100_global_shared_file/150_raw_corpora/151_librispeech/LibriSpeech/'
mkdir -p "$output_dir"

for part in ${speech_file}*/; do
    
    part_name=$(basename "$part") # 提取部分名称，如train-clean-100

    for speaker_dir in "$part"*/; do
        # 检查是否为目录
        if [[ ! -d "$speaker_dir" ]]; then
            continue
        fi
        
        speaker_id=$(basename "$speaker_dir")
        
        for chapter_dir in "$speaker_dir"*/; do
            # 检查是否为目录
            if [[ ! -d "$chapter_dir" ]]; then
                continue
            fi
            
            chapter_name=$(basename "$chapter_dir")
            transcript_file="${chapter_dir}${speaker_id}-${chapter_name}.trans.txt"
            
            # 检查转录文件是否存在
            if [[ ! -f "$transcript_file" ]]; then
                echo "Transcript file not found: $transcript_file"
                continue
            fi
            
            jsonl_file="${output_dir}/librispeech_${part_name}.jsonl"
            
            echo "Processing: $transcript_file"
            
            while IFS= read -r line; do
                # 跳过空行
                if [[ -z "$line" ]]; then
                    continue
                fi
                
                # 提取utterance_id（第一个字段）和转录文本（第二个字段到行尾）
                utterance_id=$(echo "$line" | awk '{print $1}')
                transcription=$(echo "$line" | cut -d' ' -f2-)
                
                # 构建音频文件路径
                audio_path="${chapter_dir}${utterance_id}.flac"
                
                # 检查音频文件是否存在
                if [[ ! -f "$audio_path" ]]; then
                    echo "Audio file not found: $audio_path"
                    continue
                fi
                
                # 转义文本中的特殊JSON字符
                transcription_escaped=$(echo "$transcription" | sed 's/"/\\"/g' | sed 's/\\/\\\\/g')
                
                # 写入JSONL文件
                echo "{\"key\": \"${utterance_id}\", \"source\": \"${audio_path}\", \"target\": \"${transcription_escaped}\"}" >> "$jsonl_file"
                
            done < "$transcript_file"
            
            echo "Created: $jsonl_file"
        done
    done
done

echo "JSONL files have been created in $output_dir"
