#!/usr/bin/env python3
"""
Model configurations for benchmarking different transformer sizes.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for transformer model architecture."""

    name: str
    num_layers: int
    num_heads: int
    d_model: int
    d_ff: int

    def __str__(self) -> str:
        return f"{self.name} ({self.num_layers}L-{self.num_heads}H-{self.d_model}D)"


def get_default_model_configs() -> list[ModelConfig]:
    """Get default model configurations for benchmarking based on assignment specifications."""
    return [
        ModelConfig(name="small", num_layers=12, num_heads=12, d_model=768, d_ff=3072),
        ModelConfig(name="medium", num_layers=24, num_heads=16, d_model=1024, d_ff=4096),
        ModelConfig(name="large", num_layers=36, num_heads=20, d_model=1280, d_ff=5120),
        ModelConfig(name="xl", num_layers=48, num_heads=25, d_model=1600, d_ff=6400),
        ModelConfig(name="2.7B", num_layers=32, num_heads=32, d_model=2560, d_ff=10240),
    ]
