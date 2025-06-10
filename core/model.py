import os
import onnxruntime
import numpy as np
from transformers import AutoTokenizer
from core.cache import MODEL_SESSIONS as _SESSIONS

MODELS = {
    "electra-small": {"file_name": "koelectra-small-v3-discriminator", "huggingface": "monologg/koelectra-small-v3-discriminator"},
    "electra-base": {"file_name": "koelectra-base-v3-discriminator", "huggingface": "monologg/koelectra-base-v3-discriminator"},
    "simcse": {"file_name": "BM-K/KoSimCSE-roberta", "huggingface": "BM-K/KoSimCSE-roberta"},
}

def get_embedding_by_model(model_key: str, text: str) -> list:
    if model_key not in MODELS:
        raise ValueError(f"지원하지 않는 모델: {model_key}")

    pid = os.getpid()
    session_key = f"{model_key}:{pid}"

    if session_key not in _SESSIONS:
        info = MODELS[model_key]
        model_path = f"./models/{info['file_name']}.onnx"
        sess_opt = onnxruntime.SessionOptions()
        sess_opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opt.intra_op_num_threads = 1
        sess_opt.inter_op_num_threads = 1

        tokenizer = AutoTokenizer.from_pretrained(info['huggingface'])
        session = onnxruntime.InferenceSession(model_path, sess_options=sess_opt)
        _SESSIONS[session_key] = {"tokenizer": tokenizer, "session": session}

    data = _SESSIONS[session_key]
    inputs = data['tokenizer'](text, return_tensors="np", padding="max_length", truncation=True, max_length=128)
    onnx_in = {inp.name: inputs[inp.name] for inp in data['session'].get_inputs() if inp.name in inputs}
    out = data['session'].run(None, onnx_in)[0]
    return out[0].tolist()
