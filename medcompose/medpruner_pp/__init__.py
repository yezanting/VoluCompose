"""Public exports, loaded only when the requested backend is used."""
from importlib import import_module

_EXPORTS = {'cal_compose_rate': ('.utils', 'cal_compose_rate'), 'cal_rate': ('.utils', 'cal_rate'), 'frame_group_fusion': ('.utils', 'frame_group_fusion'), 'token_allocation': ('.utils', 'token_allocation'), 'spatial_dynamic_compression2': ('.utils', 'spatial_dynamic_compression2'), 'Qwen2_5_VLForConditionalGeneration': ('.Qwen25VL', 'Qwen2_5_VLForConditionalGeneration'), 'Qwen3VLForConditionalGeneration': ('.Qwen3VL', 'Qwen3VLForConditionalGeneration'), 'HulumedQwen2ForCausalLM': ('.Hulu.hulumed_qwen2', 'HulumedQwen2ForCausalLM'), 'Gemma3ForConditionalGeneration': ('.MedGemma.gemma3', 'Gemma3ForConditionalGeneration')}

__all__ = list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value
