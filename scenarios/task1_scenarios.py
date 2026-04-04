"""
Task 1: Static Precision Assignment scenarios.
Each scenario defines a model architecture with layers to configure.
"""

TASK1_SCENARIOS = {
    "gpt2_small": {
        "scenario_id": "gpt2_small",
        "model_name": "GPT-2 125M",
        "memory_budget_gb": 8.0,
        "speed_target_speedup": 2.0,
        "layers": [
            {"name": "token_embedding", "layer_type": "embedding", "num_params": 38_000_000,
             "gradient_sensitivity": "high_variance", "activation_range": "unbounded"},
            {"name": "attn_blocks", "layer_type": "attention", "num_params": 28_000_000,
             "gradient_sensitivity": "medium", "activation_range": "normalized"},
            {"name": "layernorm", "layer_type": "layernorm", "num_params": 50_000,
             "gradient_sensitivity": "stable", "activation_range": "normalized"},
            {"name": "ffn_blocks", "layer_type": "ffn", "num_params": 56_000_000,
             "gradient_sensitivity": "stable", "activation_range": "bounded"},
            {"name": "lm_head", "layer_type": "output", "num_params": 38_000_000,
             "gradient_sensitivity": "high_precision_critical", "activation_range": "loss_computation"},
        ]
    },
    "bert_base": {
        "scenario_id": "bert_base",
        "model_name": "BERT-Base 110M",
        "memory_budget_gb": 4.0,
        "speed_target_speedup": 2.5,
        "layers": [
            {"name": "word_embeddings", "layer_type": "embedding", "num_params": 23_000_000,
             "gradient_sensitivity": "high_variance", "activation_range": "unbounded"},
            {"name": "self_attention", "layer_type": "attention", "num_params": 28_000_000,
             "gradient_sensitivity": "medium", "activation_range": "normalized"},
            {"name": "layer_norm", "layer_type": "layernorm", "num_params": 40_000,
             "gradient_sensitivity": "stable", "activation_range": "normalized"},
            {"name": "intermediate_ffn", "layer_type": "ffn", "num_params": 56_000_000,
             "gradient_sensitivity": "stable", "activation_range": "bounded"},
            {"name": "cls_head", "layer_type": "output", "num_params": 3_000_000,
             "gradient_sensitivity": "high_precision_critical", "activation_range": "loss_computation"},
        ]
    },
    "llama_7b": {
        "scenario_id": "llama_7b",
        "model_name": "LLaMA-2 7B",
        "memory_budget_gb": 24.0,
        "speed_target_speedup": 3.0,
        "layers": [
            {"name": "embed_tokens", "layer_type": "embedding", "num_params": 131_000_000,
             "gradient_sensitivity": "high_variance", "activation_range": "unbounded"},
            {"name": "self_attn", "layer_type": "attention", "num_params": 1_600_000_000,
             "gradient_sensitivity": "medium", "activation_range": "normalized"},
            {"name": "input_layernorm", "layer_type": "layernorm", "num_params": 260_000,
             "gradient_sensitivity": "stable", "activation_range": "normalized"},
            {"name": "mlp", "layer_type": "ffn", "num_params": 4_500_000_000,
             "gradient_sensitivity": "stable", "activation_range": "bounded"},
            {"name": "lm_head", "layer_type": "output", "num_params": 131_000_000,
             "gradient_sensitivity": "high_precision_critical", "activation_range": "loss_computation"},
        ]
    },
    "mistral_7b": {
        "scenario_id": "mistral_7b",
        "model_name": "Mistral 7B",
        "memory_budget_gb": 16.0,
        "speed_target_speedup": 3.5,
        "layers": [
            {"name": "embed_tokens", "layer_type": "embedding", "num_params": 131_000_000,
             "gradient_sensitivity": "high_variance", "activation_range": "unbounded"},
            {"name": "self_attn_gqa", "layer_type": "attention", "num_params": 1_200_000_000,
             "gradient_sensitivity": "medium", "activation_range": "normalized"},
            {"name": "rms_norm", "layer_type": "layernorm", "num_params": 260_000,
             "gradient_sensitivity": "stable", "activation_range": "normalized"},
            {"name": "mlp_silu", "layer_type": "ffn", "num_params": 4_800_000_000,
             "gradient_sensitivity": "stable", "activation_range": "bounded"},
            {"name": "lm_head", "layer_type": "output", "num_params": 131_000_000,
             "gradient_sensitivity": "high_precision_critical", "activation_range": "loss_computation"},
        ]
    },
    "phi3_mini": {
        "scenario_id": "phi3_mini",
        "model_name": "Phi-3 Mini 3.8B",
        "memory_budget_gb": 12.0,
        "speed_target_speedup": 2.5,
        "layers": [
            {"name": "embed_tokens", "layer_type": "embedding", "num_params": 98_000_000,
             "gradient_sensitivity": "high_variance", "activation_range": "unbounded"},
            {"name": "self_attn", "layer_type": "attention", "num_params": 900_000_000,
             "gradient_sensitivity": "medium", "activation_range": "normalized"},
            {"name": "rms_norm", "layer_type": "layernorm", "num_params": 192_000,
             "gradient_sensitivity": "stable", "activation_range": "normalized"},
            {"name": "mlp_gelu", "layer_type": "ffn", "num_params": 2_400_000_000,
             "gradient_sensitivity": "stable", "activation_range": "bounded"},
            {"name": "lm_head", "layer_type": "output", "num_params": 98_000_000,
             "gradient_sensitivity": "high_precision_critical", "activation_range": "loss_computation"},
        ]
    },
}
