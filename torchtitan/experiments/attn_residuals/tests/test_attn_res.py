# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.attn_residuals.attn_res import block_attn_res


class TestBlockAttnRes(unittest.TestCase):
    """Unit tests for the block_attn_res function."""

    def setUp(self):
        self.B, self.T, self.D = 2, 4, 16
        self.proj = nn.Linear(self.D, 1, bias=False)
        self.norm = nn.RMSNorm(self.D)

    def _make_blocks(self, n: int) -> list[torch.Tensor]:
        return [torch.randn(self.B, self.T, self.D) for _ in range(n)]

    def test_output_shape(self):
        """Output has same shape as partial_block [B, T, D]."""
        blocks = self._make_blocks(3)
        partial = torch.randn(self.B, self.T, self.D)
        out = block_attn_res(blocks, partial, self.proj, self.norm)
        self.assertEqual(out.shape, (self.B, self.T, self.D))

    def test_uniform_weights_on_zero_init(self):
        """With zero-init proj, output = mean of all sources."""
        nn.init.zeros_(self.proj.weight)
        blocks = self._make_blocks(3)
        partial = torch.randn(self.B, self.T, self.D)

        out = block_attn_res(blocks, partial, self.proj, self.norm)
        expected = torch.stack(blocks + [partial]).mean(dim=0)
        torch.testing.assert_close(out, expected)

    def test_gradient_flows_to_all_inputs(self):
        """Gradients reach proj.weight, norm.weight, blocks, and partial_block."""
        blocks = [
            torch.randn(self.B, self.T, self.D, requires_grad=True) for _ in range(3)
        ]
        partial = torch.randn(self.B, self.T, self.D, requires_grad=True)

        out = block_attn_res(blocks, partial, self.proj, self.norm)
        loss = out.sum()
        loss.backward()

        # proj and norm weights have gradients
        self.assertIsNotNone(self.proj.weight.grad)
        self.assertIsNotNone(self.norm.weight.grad)
        # All input tensors have gradients
        for i, b in enumerate(blocks):
            self.assertIsNotNone(b.grad, f"blocks[{i}] has no gradient")
        self.assertIsNotNone(partial.grad, "partial_block has no gradient")

    def test_single_block_plus_partial(self):
        """With 1 completed block + partial, output is weighted sum of 2 sources."""
        nn.init.zeros_(self.proj.weight)
        blocks = self._make_blocks(1)
        partial = torch.randn(self.B, self.T, self.D)

        out = block_attn_res(blocks, partial, self.proj, self.norm)
        # Uniform weights → mean of 2 tensors
        expected = (blocks[0] + partial) / 2
        torch.testing.assert_close(out, expected)

    def test_no_completed_blocks(self):
        """With 0 completed blocks, output is just the partial_block."""
        nn.init.zeros_(self.proj.weight)
        blocks: list[torch.Tensor] = []
        partial = torch.randn(self.B, self.T, self.D)

        out = block_attn_res(blocks, partial, self.proj, self.norm)
        # Only 1 source → softmax([0]) = [1.0] → output = partial_block
        torch.testing.assert_close(out, partial)

    def test_numerical_stability_large_logits(self):
        """Softmax doesn't produce NaN/Inf for large logit differences."""
        # Set proj to produce very large logits
        nn.init.constant_(self.proj.weight, 100.0)
        blocks = self._make_blocks(3)
        partial = torch.randn(self.B, self.T, self.D)

        out = block_attn_res(blocks, partial, self.proj, self.norm)
        self.assertFalse(torch.isnan(out).any(), "Output contains NaN")
        self.assertFalse(torch.isinf(out).any(), "Output contains Inf")

    def test_deterministic_output(self):
        """Same inputs produce same outputs across calls."""
        torch.manual_seed(42)
        blocks = self._make_blocks(3)
        partial = torch.randn(self.B, self.T, self.D)

        out1 = block_attn_res(blocks, partial, self.proj, self.norm)
        out2 = block_attn_res(blocks, partial, self.proj, self.norm)
        torch.testing.assert_close(out1, out2)

    def test_output_dtype_matches_input(self):
        """Output dtype matches input dtype."""
        blocks = self._make_blocks(2)
        partial = torch.randn(self.B, self.T, self.D)
        out = block_attn_res(blocks, partial, self.proj, self.norm)
        self.assertEqual(out.dtype, partial.dtype)


if __name__ == "__main__":
    unittest.main()
