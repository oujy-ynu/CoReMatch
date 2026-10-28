"""Dataset defaults and model settings."""

DATASET_CONFIGS = {
    "AIDS700nef": dict(anchor_count=16, anchor_dim=32, seed=1, epochs=30000),
    "LINUX": dict(anchor_count=8, anchor_dim=16, seed=3, epochs=13000),
    "IMDBMulti": dict(anchor_count=6, anchor_dim=16, seed=2, epochs=10000),
}


ARCHITECTURE = dict(residual_alpha=0.1, dropout=0.0, attention_heads=4,
                    ffn_ratio=2, attention_dropout=0.0, layer_scale_init=0.1)
