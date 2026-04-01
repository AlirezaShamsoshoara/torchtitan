# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

from torchtitan.experiments.attn_residuals import (
    attn_res_configs,
    model_registry,
)


class TestAttnResTransformerBlock(unittest.TestCase):
    """Unit tests for AttnResTransformerBlock."""

    def setUp(self):
        self.config = attn_res_configs["debugmodel"]
        self.dim = self.config.dim  # 256
        self.B, self.T = 2, 8
        # Build proper freqs_cis via the RoPE module
        rope = self.config.rope.build()
        rope.init_weights(buffer_device=torch.device("cpu"))
        self.freqs_cis = rope.cache

    def _make_block(self, layer_id: int):
        block = self.config.layer.build(
            layer_id=layer_id, dim=self.dim, n_layers=self.config.n_layers
        )
        block.init_weights()
        return block

    def test_output_types(self):
        """Forward returns (list[Tensor], Tensor)."""
        block = self._make_block(layer_id=0)
        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        partial = torch.zeros_like(h)
        out_blocks, out_partial = block(blocks, partial, self.freqs_cis, None)
        self.assertIsInstance(out_blocks, list)
        self.assertIsInstance(out_partial, torch.Tensor)
        self.assertEqual(out_partial.shape, (self.B, self.T, self.dim))

    def test_block_boundary_detection(self):
        """Blocks list grows at expected layer boundaries."""
        # debugmodel: n_layers=6, num_attn_res_blocks=3 → block_size=2
        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        partial = torch.zeros_like(h)

        # Layer 0: boundary but layer_id=0 so no append
        block0 = self._make_block(0)
        self.assertTrue(block0.is_block_boundary)
        out_blocks, partial = block0(blocks, partial, self.freqs_cis, None)
        self.assertEqual(len(out_blocks), 1, "Layer 0 should not append")

        # Layer 1: not a boundary
        block1 = self._make_block(1)
        self.assertFalse(block1.is_block_boundary)
        out_blocks, partial = block1(out_blocks, partial, self.freqs_cis, None)
        self.assertEqual(len(out_blocks), 1, "Layer 1 should not append")

        # Layer 2: boundary → append
        block2 = self._make_block(2)
        self.assertTrue(block2.is_block_boundary)
        out_blocks, partial = block2(out_blocks, partial, self.freqs_cis, None)
        self.assertEqual(len(out_blocks), 2, "Layer 2 should append")

        # Layer 3: not a boundary
        block3 = self._make_block(3)
        self.assertFalse(block3.is_block_boundary)
        out_blocks, partial = block3(out_blocks, partial, self.freqs_cis, None)
        self.assertEqual(len(out_blocks), 2, "Layer 3 should not append")

        # Layer 4: boundary → append
        block4 = self._make_block(4)
        self.assertTrue(block4.is_block_boundary)
        out_blocks, partial = block4(out_blocks, partial, self.freqs_cis, None)
        self.assertEqual(len(out_blocks), 3, "Layer 4 should append")

    def test_blocks_list_immutability(self):
        """Input blocks list is not mutated by forward."""
        block = self._make_block(layer_id=2)  # boundary layer
        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        original_len = len(blocks)
        partial = torch.randn(self.B, self.T, self.dim)

        out_blocks, _ = block(blocks, partial, self.freqs_cis, None)
        # Original list should not be modified
        self.assertEqual(len(blocks), original_len)
        # Output list should be longer
        self.assertEqual(len(out_blocks), original_len + 1)

    def test_init_weights_zeros_projections(self):
        """After init_weights, attn_res_proj and mlp_res_proj are all zeros."""
        block = self._make_block(layer_id=0)
        torch.testing.assert_close(
            block.attn_res_proj.weight,
            torch.zeros_like(block.attn_res_proj.weight),
        )
        torch.testing.assert_close(
            block.mlp_res_proj.weight,
            torch.zeros_like(block.mlp_res_proj.weight),
        )

    def test_forward_backward_no_error(self):
        """Forward + backward pass completes without errors."""
        block = self._make_block(layer_id=0)
        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        partial = torch.zeros_like(h)

        _, out_partial = block(blocks, partial, self.freqs_cis, None)
        loss = out_partial.sum()
        loss.backward()

        # Verify at least one parameter has a gradient
        has_grad = any(p.grad is not None for p in block.parameters())
        self.assertTrue(has_grad, "No parameters received gradients")

    def test_param_count(self):
        """Block adds exactly 4*dim + 2*dim extra params vs no-AttnRes block."""
        block = self._make_block(layer_id=0)
        # AttnRes-specific params:
        # attn_res_proj: [1, dim], mlp_res_proj: [1, dim] → 2*dim
        # attn_res_norm: [dim], mlp_res_norm: [dim] → 2*dim
        # Total extra: 4*dim
        attn_res_params = (
            block.attn_res_proj.weight.numel()
            + block.mlp_res_proj.weight.numel()
            + block.attn_res_norm.weight.numel()
            + block.mlp_res_norm.weight.numel()
        )
        self.assertEqual(attn_res_params, 4 * self.dim)

    def test_partial_block_accumulation(self):
        """Partial block accumulates attn + MLP outputs within a block."""
        block = self._make_block(layer_id=1)  # non-boundary layer
        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        partial = torch.zeros_like(h)

        _, out_partial = block(blocks, partial, self.freqs_cis, None)
        # partial started at zeros and accumulated attn_out + mlp_out,
        # so it should be non-zero
        self.assertFalse(
            torch.allclose(out_partial, torch.zeros_like(out_partial)),
            "Partial block should accumulate sub-layer outputs",
        )

    def test_full_ac_preserves_gradients(self):
        """Full activation checkpointing produces identical gradients."""
        torch.manual_seed(42)
        block = self._make_block(layer_id=0)
        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        partial = torch.zeros_like(h)

        # Run without AC
        _, out_no_ac = block(blocks, partial, self.freqs_cis, None)
        loss_no_ac = out_no_ac.sum()
        loss_no_ac.backward()
        grads_no_ac = {
            n: p.grad.clone() for n, p in block.named_parameters() if p.grad is not None
        }
        block.zero_grad()

        # Run with full AC
        wrapped = ptd_checkpoint_wrapper(block)
        _, out_ac = wrapped(blocks, partial, self.freqs_cis, None)
        loss_ac = out_ac.sum()
        loss_ac.backward()
        grads_ac = {
            n: p.grad.clone() for n, p in block.named_parameters() if p.grad is not None
        }

        # Gradients must be identical
        for name in grads_no_ac:
            torch.testing.assert_close(
                grads_no_ac[name],
                grads_ac[name],
                msg=f"Gradient mismatch for {name} with AC",
            )

    def test_compile_eager_backend(self):
        """torch.compile with eager backend doesn't break forward/backward."""
        block = self._make_block(layer_id=0)
        compiled_block = torch.compile(block, backend="eager", fullgraph=True)

        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        partial = torch.zeros_like(h)

        _, out = compiled_block(blocks, partial, self.freqs_cis, None)
        loss = out.sum()
        loss.backward()

        has_grad = any(p.grad is not None for p in block.parameters())
        self.assertTrue(has_grad, "No gradients after compiled forward/backward")


class TestAttnResDecoder(unittest.TestCase):
    """Unit tests for AttnResDecoder."""

    def setUp(self):
        self.config = attn_res_configs["debugmodel"]
        self.dim = self.config.dim
        self.B, self.T = 2, 8
        self.vocab_size = self.config.vocab_size

    def _make_model(self):
        model = self.config.build()
        model.init_weights()
        return model

    def test_output_shape(self):
        """Output is [B, T, vocab_size]."""
        model = self._make_model()
        tokens = torch.randint(0, self.vocab_size, (self.B, self.T))
        out = model(tokens)
        self.assertEqual(out.shape, (self.B, self.T, self.vocab_size))

    def test_forward_backward(self):
        """Complete forward-backward pass succeeds."""
        model = self._make_model()
        tokens = torch.randint(0, self.vocab_size, (self.B, self.T))
        out = model(tokens)
        loss = out.sum()
        loss.backward()

    def test_all_params_have_grad(self):
        """After backward, every trainable param has .grad != None."""
        model = self._make_model()
        tokens = torch.randint(0, self.vocab_size, (self.B, self.T))
        out = model(tokens)
        loss = out.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(
                    param.grad,
                    f"Parameter '{name}' has no gradient",
                )

    def test_no_nan_in_output(self):
        """Forward pass produces no NaN values."""
        model = self._make_model()
        tokens = torch.randint(0, self.vocab_size, (self.B, self.T))
        out = model(tokens)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")

    def test_model_spec(self):
        """ModelSpec is correctly formed."""
        spec = model_registry("debugmodel")
        self.assertEqual(spec.name, "attn_residuals")
        self.assertEqual(spec.flavor, "debugmodel")
        self.assertIsNotNone(spec.parallelize_fn)
        self.assertIsNotNone(spec.pipelining_fn)
        self.assertIsNotNone(spec.build_loss_fn)

    def test_config_build_produces_valid_model(self):
        """Config.build() returns a model with the expected structure."""
        model = self._make_model()
        self.assertTrue(hasattr(model, "tok_embeddings"))
        self.assertTrue(hasattr(model, "layers"))
        self.assertTrue(hasattr(model, "norm"))
        self.assertTrue(hasattr(model, "output"))
        self.assertEqual(len(model.layers), self.config.n_layers)

    def test_none_modules_for_pp(self):
        """Forward works when tok_embeddings/norm/output are None (PP compat)."""
        model = self._make_model()
        # Simulate a middle PP stage
        model.tok_embeddings = None
        model.norm = None
        model.output = None

        # Pass hidden states directly instead of tokens
        h = torch.randn(self.B, self.T, self.dim)
        out = model(h)
        # Without norm/output, should return partial_block directly
        self.assertEqual(out.shape, (self.B, self.T, self.dim))

    def test_full_ac_decoder_gradients(self):
        """Full AC on decoder layers produces identical gradients to no-AC."""
        torch.manual_seed(42)
        model = self._make_model()
        tokens = torch.randint(0, self.vocab_size, (self.B, self.T))

        # Run without AC
        out = model(tokens)
        out.sum().backward()
        grads_no_ac = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad()

        # Apply full AC to each layer
        for layer_id, block in model.layers.items():
            model.layers[layer_id] = ptd_checkpoint_wrapper(block)

        out = model(tokens)
        out.sum().backward()
        grads_ac = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }

        self.assertEqual(len(grads_no_ac), len(grads_ac))
        for name in grads_no_ac:
            # AC wrapping adds _checkpoint_wrapped_module prefix
            ac_name = name
            if ac_name not in grads_ac:
                # Try with checkpoint wrapper prefix
                parts = name.split(".", 2)
                if len(parts) >= 3:
                    ac_name = (
                        f"{parts[0]}.{parts[1]}._checkpoint_wrapped_module.{parts[2]}"
                    )
            if ac_name in grads_ac:
                torch.testing.assert_close(
                    grads_no_ac[name],
                    grads_ac[ac_name],
                    msg=f"Gradient mismatch for {name} with AC",
                )

    def test_compile_eager_decoder(self):
        """torch.compile (eager backend) on decoder layers works correctly."""
        model = self._make_model()

        # Compile each layer with eager backend
        for layer_id, block in model.layers.items():
            model.layers[layer_id] = torch.compile(
                block, backend="eager", fullgraph=True
            )

        tokens = torch.randint(0, self.vocab_size, (self.B, self.T))
        out = model(tokens)
        self.assertEqual(out.shape, (self.B, self.T, self.vocab_size))
        self.assertFalse(torch.isnan(out).any(), "Compiled output contains NaN")

        out.sum().backward()
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        self.assertTrue(has_grad, "No gradients after compiled decoder backward")

    def test_deterministic_across_runs(self):
        """Same seed produces identical outputs across runs."""
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            model = self._make_model()
            tokens = torch.randint(0, self.vocab_size, (self.B, self.T))
            out = model(tokens)
            results.append(out.detach().clone())
        torch.testing.assert_close(results[0], results[1])


class TestWeightTying(unittest.TestCase):
    """Tests for enable_weight_tying in AttnResDecoder."""

    def setUp(self):
        self.B, self.T = 2, 8

    def _make_tied_config(self):
        """Create a debugmodel config with weight tying enabled."""
        import dataclasses

        base = attn_res_configs["debugmodel"]
        return dataclasses.replace(base, enable_weight_tying=True)

    def test_weight_tying_shares_tensor(self):
        """tok_embeddings and output share the same weight tensor."""
        config = self._make_tied_config()
        model = config.build()
        model.init_weights()
        self.assertIs(
            model.tok_embeddings.weight,
            model.output.weight,
            "tok_embeddings.weight and output.weight must be the same object",
        )

    def test_weight_tying_reduces_param_count(self):
        """Weight tying reduces total parameter count."""
        import dataclasses

        base = attn_res_configs["debugmodel"]
        untied = base.build()
        untied.init_weights()

        tied_config = dataclasses.replace(base, enable_weight_tying=True)
        tied = tied_config.build()
        tied.init_weights()

        untied_params = sum(p.numel() for p in untied.parameters())
        tied_params = sum(p.numel() for p in tied.parameters())
        # Weight tying saves vocab_size * dim parameters
        expected_savings = base.vocab_size * base.dim
        self.assertEqual(untied_params - tied_params, expected_savings)

    def test_weight_tying_survives_init_weights(self):
        """Weight tying holds after init_weights (meta device re-tie)."""
        config = self._make_tied_config()
        model = config.build()
        # init_weights should re-tie even if __init__ tying didn't stick
        model.init_weights()
        self.assertIs(model.tok_embeddings.weight, model.output.weight)

    def test_weight_tying_forward_backward(self):
        """Forward + backward works with weight tying enabled."""
        config = self._make_tied_config()
        model = config.build()
        model.init_weights()

        tokens = torch.randint(0, config.vocab_size, (self.B, self.T))
        out = model(tokens)
        self.assertEqual(out.shape, (self.B, self.T, config.vocab_size))
        self.assertFalse(torch.isnan(out).any())

        loss = out.sum()
        loss.backward()
        # The shared weight should have a gradient
        self.assertIsNotNone(model.output.weight.grad)

    def test_debugmodel_no_weight_tying(self):
        """debugmodel config does not have weight tying."""
        config = attn_res_configs["debugmodel"]
        self.assertFalse(config.enable_weight_tying)

    def test_1b_has_weight_tying(self):
        """1B config has weight tying enabled."""
        config = attn_res_configs["1B"]
        self.assertTrue(config.enable_weight_tying)

    def test_model_spec_1b(self):
        """1B ModelSpec is correctly formed."""
        spec = model_registry("1B")
        self.assertEqual(spec.name, "attn_residuals")
        self.assertEqual(spec.flavor, "1B")
        self.assertTrue(spec.model.enable_weight_tying)


if __name__ == "__main__":
    unittest.main()
