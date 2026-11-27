"""modsae: sparse autoencoders on the grokked modular-addition transformer.

Global context: this package trains the 1-layer modular-addition transformer of
Nanda et al. 2023 (https://arxiv.org/abs/2301.05217), trains SAEs on its activations
with SAELens (https://github.com/decoderesearch/SAELens), characterizes SAE features
against the known Fourier mechanism, and runs steering experiments. The CLI wrapper
in `modsae.cli` calls each phase as an imported function.
"""
