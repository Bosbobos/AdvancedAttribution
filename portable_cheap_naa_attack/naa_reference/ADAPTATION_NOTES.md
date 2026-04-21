# NAA Adaptation Notes

This directory contains a paper-faithful NAA baseline adapted to the self-contained `portable_cheap_naa_attack` package.

## What was kept close to the original algorithm

- zero baseline `x' = 0`
- aggregate neuron-importance gradients over `N=30` scaled inputs
- feature-space loss `((y - y') * agg_grad).sum()`
- momentum iterative update under an `L_inf` budget

## What had to be changed for our setup

- The original repository was not directly accessible from the runtime environment, so the implementation was matched against the paper and a source-level NAA reimplementation from `torchattack`.
- Model loading was adapted from the original ImageNet-style backbones to `ultralytics` `yolo11s-cls`.
- Feature-layer selection was generalized to arbitrary `model.named_modules()` strings such as `model.6`.
- Input preprocessing was replaced with the same self-contained letterbox loader used by the cheap-IG package.
- Result tracking, PNG export, JSON export, and batch diagnostics were aligned with the existing portable package API.
- Optional `variant="pd"` support reuses the same DIM/PIM wrappers as the cheap-IG method for fair side-by-side experiments, even though the default NAA baseline remains `variant="base"`.
