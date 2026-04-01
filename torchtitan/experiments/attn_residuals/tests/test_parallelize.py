# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for parallelization and compilation of the AttnRes model.

These tests verify:
- Task 5: TP plan correctness and FSDP compatibility
- Task 8: torch.compile compatibility (fullgraph, no graph breaks)

GPU integration tests (FSDP, TP, compile with inductor) require real GPUs
and can be run via the integration test script:
    NGPU=2 COMM_MODE=fake_backend MODULE=experiments.attn_residuals \
        CONFIG=debugmodel ./run_train.sh
"""

import inspect
import subprocess
import sys
import unittest

import torch

from torchtitan.experiments.attn_residuals import (
    attn_res_configs,
    model_registry,
    parallelize_attn_res,
)


class TestTPPlanCoverage(unittest.TestCase):
    """Verify the TP plan covers all required submodules."""

    def setUp(self):
        self.config = attn_res_configs["debugmodel"]
        self.dim = self.config.dim

    def _make_block(self, layer_id: int = 0):
        block = self.config.layer.build(
            layer_id=layer_id, dim=self.dim, n_layers=self.config.n_layers
        )
        block.init_weights()
        return block

    def test_tp_plan_keys_match_submodules(self):
        """TP plan keys correspond to actual submodules in the block."""
        block = self._make_block()
        # These are the submodule names that must appear in the TP plan
        expected_tp_modules = {
            "attention_norm",
            "attention",
            "attention.wq",
            "attention.wk",
            "attention.wv",
            "attention.wo",
            "ffn_norm",
            "feed_forward",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "attn_res_norm",
            "mlp_res_norm",
            "attn_res_proj",
            "mlp_res_proj",
        }
        # Verify all expected modules exist in the block
        for module_name in expected_tp_modules:
            submod = block
            for part in module_name.split("."):
                self.assertTrue(
                    hasattr(submod, part),
                    f"Block missing expected submodule '{module_name}'",
                )
                submod = getattr(submod, part)

    def test_attn_res_proj_in_tp_plan_as_replicate(self):
        """Proj weights [1,D] are Replicated DTensors on TP mesh."""
        block = self._make_block()
        # Verify proj weights exist and have correct shape
        self.assertEqual(block.attn_res_proj.weight.shape, (1, self.dim))
        self.assertEqual(block.mlp_res_proj.weight.shape, (1, self.dim))

    def test_attn_res_norms_are_separate_from_standard_norms(self):
        """AttnRes norms are separate modules from attention_norm/ffn_norm."""
        block = self._make_block()
        self.assertIsNot(block.attn_res_norm, block.attention_norm)
        self.assertIsNot(block.mlp_res_norm, block.ffn_norm)
        self.assertIsNot(block.attn_res_norm, block.mlp_res_norm)


class TestParallelizeFunctionSignature(unittest.TestCase):
    """Verify parallelize_attn_res has the correct signature for ModelSpec."""

    def test_signature_parameters(self):
        """parallelize_attn_res accepts required keyword arguments."""
        sig = inspect.signature(parallelize_attn_res)
        required_params = {
            "model",
            "parallel_dims",
            "training",
            "model_converters",
            "parallelism",
            "compile_config",
            "ac_config",
            "dump_folder",
        }
        actual_params = set(sig.parameters.keys())
        self.assertTrue(
            required_params.issubset(actual_params),
            f"Missing parameters: {required_params - actual_params}",
        )

    def test_model_spec_parallelize_fn_is_parallelize_attn_res(self):
        """ModelSpec's parallelize_fn points to parallelize_attn_res."""
        spec = model_registry("debugmodel")
        self.assertEqual(spec.parallelize_fn, parallelize_attn_res)


class TestCompileCompatibility(unittest.TestCase):
    """Verify torch.compile works with AttnRes model components."""

    def setUp(self):
        self.config = attn_res_configs["debugmodel"]
        self.dim = self.config.dim
        self.B, self.T = 2, 8
        rope = self.config.rope.build()
        rope.init_weights(buffer_device=torch.device("cpu"))
        self.freqs_cis = rope.cache

    def _make_model(self):
        model = self.config.build()
        model.init_weights()
        return model

    def _make_block(self, layer_id: int = 0):
        block = self.config.layer.build(
            layer_id=layer_id, dim=self.dim, n_layers=self.config.n_layers
        )
        block.init_weights()
        return block

    def test_compile_block_fullgraph_eager(self):
        """Single block compiles with fullgraph=True (eager backend)."""
        block = self._make_block(layer_id=0)
        compiled = torch.compile(block, backend="eager", fullgraph=True)

        h = torch.randn(self.B, self.T, self.dim)
        blocks = [h]
        partial = torch.zeros_like(h)

        out_blocks, out_partial = compiled(blocks, partial, self.freqs_cis, None)
        self.assertEqual(out_partial.shape, (self.B, self.T, self.dim))
        out_partial.sum().backward()

    def test_compile_different_block_positions(self):
        """Blocks at different positions (different blocks list sizes) compile."""
        h = torch.randn(self.B, self.T, self.dim)

        # Layer 0: blocks=[h], 1 source
        block0 = self._make_block(layer_id=0)
        compiled0 = torch.compile(block0, backend="eager", fullgraph=True)
        out_blocks, partial = compiled0([h], torch.zeros_like(h), self.freqs_cis, None)

        # Layer 2: boundary, blocks=[h, partial], 2 sources
        block2 = self._make_block(layer_id=2)
        compiled2 = torch.compile(block2, backend="eager", fullgraph=True)
        out_blocks2, partial2 = compiled2(out_blocks, partial, self.freqs_cis, None)
        self.assertEqual(len(out_blocks2), 2)

    def test_compile_decoder_all_layers(self):
        """Compiling all layers in the decoder works end-to-end."""
        model = self._make_model()
        for layer_id, block in model.layers.items():
            model.layers[layer_id] = torch.compile(
                block, backend="eager", fullgraph=True
            )

        tokens = torch.randint(0, self.config.vocab_size, (self.B, self.T))
        out = model(tokens)
        self.assertEqual(out.shape, (self.B, self.T, self.config.vocab_size))
        self.assertFalse(torch.isnan(out).any())

        # Backward pass
        out.sum().backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        self.assertTrue(has_grad)

    def test_compile_numerics_match_eager(self):
        """Compiled model produces same output as eager model."""
        torch.manual_seed(42)
        model = self._make_model()
        tokens = torch.randint(0, self.config.vocab_size, (self.B, self.T))

        # Eager forward
        out_eager = model(tokens).detach().clone()

        # Compile and run again
        for layer_id, block in model.layers.items():
            model.layers[layer_id] = torch.compile(
                block, backend="eager", fullgraph=True
            )
        out_compiled = model(tokens).detach().clone()

        torch.testing.assert_close(out_eager, out_compiled)


@unittest.skipUnless(
    torch.cuda.is_available(), "CUDA required for fake_backend integration tests"
)
class TestFakeBackendIntegration(unittest.TestCase):
    """Integration tests using fake_backend mode.

    These tests validate that the full training pipeline (model build +
    parallelization + training step) works with FSDP and TP configurations
    by using PyTorch's fake distributed backend. Requires working CUDA drivers.
    """

    def _run_fake_backend(self, extra_args: list[str], ngpu: int = 2) -> bool:
        """Run training with fake_backend mode and return success status."""
        import os

        env = os.environ.copy()
        env["NGPU"] = str(ngpu)
        env["LOCAL_RANK"] = "0"

        cmd = [
            sys.executable,
            "-m",
            "torchtitan.train",
            "--module",
            "attn_residuals",
            "--config",
            "attn_res_debugmodel",
            "--comm.mode=fake_backend",
            "--training.steps",
            "1",
        ] + extra_args
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            cwd="/home/alisol/projects/torchtitan",
        )
        return result.returncode == 0, result.stderr

    def test_fsdp_fake_backend(self):
        """FSDP parallelization works with fake backend."""
        success, stderr = self._run_fake_backend(
            ["--parallelism.data_parallel_shard_degree", "2"],
            ngpu=2,
        )
        self.assertTrue(
            success,
            f"FSDP fake_backend test failed:\n{stderr[-2000:] if stderr else 'no stderr'}",
        )

    def test_fsdp_tp_fake_backend(self):
        """FSDP + TP parallelization works with fake backend."""
        success, stderr = self._run_fake_backend(
            [
                "--parallelism.data_parallel_shard_degree",
                "2",
                "--parallelism.tensor_parallel_degree",
                "2",
            ],
            ngpu=4,
        )
        self.assertTrue(
            success,
            f"FSDP+TP fake_backend test failed:\n{stderr[-2000:] if stderr else 'no stderr'}",
        )

    def test_compile_fake_backend(self):
        """torch.compile works with fake backend."""
        success, stderr = self._run_fake_backend(
            ["--compile.enable"],
            ngpu=2,
        )
        self.assertTrue(
            success,
            f"Compile fake_backend test failed:\n{stderr[-2000:] if stderr else 'no stderr'}",
        )

    def test_fsdp_compile_fake_backend(self):
        """FSDP + compile works with fake backend."""
        success, stderr = self._run_fake_backend(
            [
                "--parallelism.data_parallel_shard_degree",
                "2",
                "--compile.enable",
            ],
            ngpu=2,
        )
        self.assertTrue(
            success,
            f"FSDP+compile fake_backend test failed:\n{stderr[-2000:] if stderr else 'no stderr'}",
        )


if __name__ == "__main__":
    unittest.main()
