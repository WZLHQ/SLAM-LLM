1. asr_config.py中的ModelConfig缺少whisper_decode与encoder_path_hf的定义，我做了如下添加
    @dataclass
    class ModelConfig:
        file: str = "examples/asr_librispeech/model/slam_model_asr.py:model_factory"
        llm_name: str = "vicuna-13b-v1.5"
        llm_path: str = "PATH/to/LLAMA/7B"
        llm_type: str = "decoder_only"
        llm_dim: int = 4096

        whisper_decode : Optional[bool] = False # added by QH
        encoder_path_hf: Optional[str] = None # added by QH

2. 代码中的train_config.use_fp16没有起到实质作用，我的更改如下（在pipeline.finetune中）
    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    model=model.to(torch.bfloat16) # added by QH


3. 一些库没有安装上，
    soundfile用pip安装即可
    ffmpeg用这个语句安装：apt-get update && apt-get install ffmpeg

4. with MemoryTrace() as memtrace,Join([model]):  # track the memory usage会导致如下错误：
    AttributeError: 'slam_model_asr' object has no attribute 'join_hook'
    按照链接（https://github.com/X-LANCE/SLAM-LLM/issues/231）中的方式解决即可



