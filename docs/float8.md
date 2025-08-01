## Float8 Training on H100s

Float8 training can provide substantial training speedups for models where the majority of GEMMs are sufficiently large enough that the speedup from using float8 tensorcores outweighs the overhead of dynamic quantization. See [here](https://github.com/pytorch/ao/tree/main/torchao/float8#performance) for microbenchmarks detailing the observed speedups for the forward + backward pass of a simple "layer norm => linear => sigmoid" model for different M,N,K sizes, which can help you determine if your model can benefit from float8 training. Note you can also use float8 training for only the subset of layers which will benefit from it by using the [filter_fqns](https://github.com/pytorch/torchtitan/blob/3b85aa31fffc46ecbf785a57ee314a01614f572f/torchtitan/config_manager.py#L448) argument.

### Usage steps

Please install latest [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/float8) to support float8 dtype
```
USE_CPP=0 python -m pip install git+https://github.com/pytorch/ao.git
```

For float8 with tensorwise scaling, launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --model.converters="float8" --float8.enable_fsdp_float8_all_gather --float8.precompute_float8_dynamic_scale_for_fsdp --float8.force_recompute_fp8_weight_in_bwd --training.compile
```
* `--model.converters="float8"`: swap `nn.Linear` with `Float8Linear` to perform float8 matmul.
* `--float8.enable_fsdp_float8_all_gather`: cast `Float8Linear.weight` from high precision to float8 before FSDP all-gather so we can communicate in float8 to save bandwidth.
* `--float8.precompute_float8_dynamic_scale_for_fsdp` (optional): communicate AMAX/scales efficiently in a single all-reduce for all parameters instead of doing many small all-reduce for each parameter.
* `--float8.force_recompute_fp8_weight_in_bwd` (optional): force recomputation of fp8 weights during backward pass, preventing unsharded fp8 weights from being saved for backward.
* `--float8.filter_fqns="..."` (optional): a comma separated list of fully qualified names of modules not to convert to float8 training. Example: `--float8.filter_fqns="attention.wk,attention.wv"`. You can determine which layers to convert by looking at the microbenchmarks in the [performance section](https://github.com/pytorch/ao/tree/main/torchao/float8#performance) of the torchao documentation for the float8 recipe you're using.
    * **Auto-filter**: add `"auto_filter_small_kn"` as one of the `--float8.filter_fqns=...` to to enable automatic module filtering, which will automatically not convert linear layers are not large enough to benefit from float8 training, since the GEMM has to be big enough that the speedup from using FP8 tensorcores is greater than the overhead of creating dynamically quantized inputs. The thresholds for conversion are based on microbenchmarks measured on NVIDIA H100 GPUs, where (K,N) represents the linear layer weight shape. For best performance, you should still manually filter out layers that are too small to benefit from float8 training.
* `--training.compile` (required for competitive performance): use `torch.compile` to fuse the float8 scaling/casting kernels

For float8 with rowwise scaling, launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --model.converters="float8" --float8.recipe_name rowwise --training.compile
```
* `--model.converters="float8"`: swap `nn.Linear` with `Float8Linear` to perform float8 matmul.
* `--float8.recipe_name="rowwise"`: use the rowwise scaling recipe for higher accuracy compared to tensorwise scaling
* `--training.compile` (required for competitive performance): use `torch.compile` to fuse the float8 scaling/casting kernels

For parallelisms, for float8 with tensorwise scaling we support float8 all-gather for FSDP (optional) and for TP (by default for `Float8Linear`). For float8 with rowwise scaling, all distributed communication is done in high precision.

For scaling strategy, we currently support tensorwise dynamic scaling (stable) and rowwise dynamic scaling (alpha).

## Benefits of composing of float8 with tensorwise scaling with `torch.distributed`
**Float8 vs Bfloat16/Float32**: In float8 E4M3 format, we only have 3 bits for mantissa, it becomes user's responsibility to maintain consistent scales across operations (summation, multiplication) to balance between precision and range. For bfloat16/float32, exponent range is large enough and users do not need to maintain such scales. When using float8 in FSDP and TP, tensors are sharded across ranks. To keep single device semantics, it's critical to communicate scales across ranks.

As shown below, for float8 for matmul, `torch._scaled_mm` requires both float8 tensors and their scales. Scales are calculated from `max(abs)` of a high precision tensor.
```
# float32/bfloat16 matmul, `torch.mm(input, weight)`, does not require scales
# float8 matmul requires scales to ensure values to fit within the representable range
torch._scaled_mm(input_fp8, weight_fp8, scale_a=scale_input, scale_b=scale_weight)
```

For single device training, we cast input and weight into float8 inside forward before calling `torch._scaled_mm`.

For FSDP, weights are sharded across ranks. We cast high precision weights (1/N on each rank) into float8, and perform float8 all-gather to save bandwidth. At the beginning of the forward, we already have the unsharded float8 weights. The overhead is communicating `max(abs)` across ranks. Float8 all-gather and amax communication can be a net win over float32/bfloat16 all-gather, depending on world size and message size.

For TP, a typical example is row-wise sharded input and column-wise sharded weight. For input, we cast sharded input into float8 and perform float8 all-gather for unsharded input. The overhead is communicating `max(abs)` across ranks. For sharded weights, we communicate `max(abs)` as well. Inside the forward, we perform matmul with float8 input (unsharded) and float8 weight (sharded) with their global `max(abs)`.
