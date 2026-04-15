# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

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


    def test_batched_norm_matches_per_source_loop(self):
        """Batched F.rms_norm produces identical results to per-source nn.RMSNorm loop.

        This verifies the Task 15 optimization: replacing the per-source norm
        loop with a single batched F.rms_norm call does not change the output.
        """
        torch.manual_seed(123)
        B, T, D = 4, 32, 128
        N = 9  # 8 blocks + 1 partial (typical for num_attn_res_blocks=8)
        proj = nn.Linear(D, 1, bias=False)
        norm = nn.RMSNorm(D)

        blocks = [torch.randn(B, T, D) for _ in range(N - 1)]
        partial = torch.randn(B, T, D)
        sources = blocks + [partial]

        # Method 1: per-source loop (old implementation)
        w = proj.weight  # [1, D]
        logits_loop = torch.stack([(norm(v) * w).sum(dim=-1) for v in sources])
        weights_loop = logits_loop.softmax(dim=0)
        V_loop = torch.stack(sources)
        out_loop = (weights_loop.unsqueeze(-1) * V_loop).sum(dim=0)

        # Method 2: batched F.rms_norm (new implementation)
        V = torch.stack(sources)
        K = F.rms_norm(V, norm.normalized_shape, norm.weight, norm.eps)
        logits_batch = (K * proj.weight).sum(dim=-1)
        weights_batch = logits_batch.softmax(dim=0)
        out_batch = (weights_batch.unsqueeze(-1) * V).sum(dim=0)

        # Must be bitwise identical (same math, same order of ops)
        torch.testing.assert_close(logits_loop, logits_batch, atol=0, rtol=0)
        torch.testing.assert_close(out_loop, out_batch, atol=0, rtol=0)

    def test_batched_norm_matches_varying_source_counts(self):
        """Batched norm matches per-source loop for 1, 2, 5, and 16 sources."""
        torch.manual_seed(456)
        D = 64
        proj = nn.Linear(D, 1, bias=False)
        norm = nn.RMSNorm(D)

        for n_sources in [1, 2, 5, 16]:
            sources = [torch.randn(2, 8, D) for _ in range(n_sources)]

            # Per-source loop
            w = proj.weight
            logits_loop = torch.stack([(norm(v) * w).sum(dim=-1) for v in sources])

            # Batched
            V = torch.stack(sources)
            K = F.rms_norm(V, norm.normalized_shape, norm.weight, norm.eps)
            logits_batch = (K * proj.weight).sum(dim=-1)

            torch.testing.assert_close(
                logits_loop, logits_batch, atol=0, rtol=0,
                msg=f"Mismatch with {n_sources} sources",
            )

    def test_batched_norm_gradient_equivalence(self):
        """Gradients through batched norm match per-source loop gradients."""
        torch.manual_seed(789)
        D = 64
        proj1 = nn.Linear(D, 1, bias=False)
        norm1 = nn.RMSNorm(D)
        proj2 = nn.Linear(D, 1, bias=False)
        norm2 = nn.RMSNorm(D)

        # Copy weights so both start identical
        proj2.weight.data.copy_(proj1.weight.data)
        norm2.weight.data.copy_(norm1.weight.data)

        sources_data = [torch.randn(2, 8, D) for _ in range(5)]

        # Per-source loop path
        srcs1 = [s.clone().requires_grad_(True) for s in sources_data]
        w1 = proj1.weight
        logits1 = torch.stack([(norm1(v) * w1).sum(dim=-1) for v in srcs1])
        weights1 = logits1.softmax(dim=0)
        V1 = torch.stack(srcs1)
        out1 = (weights1.unsqueeze(-1) * V1).sum(dim=0)
        out1.sum().backward()

        # Batched path
        srcs2 = [s.clone().requires_grad_(True) for s in sources_data]
        V2 = torch.stack(srcs2)
        K2 = F.rms_norm(V2, norm2.normalized_shape, norm2.weight, norm2.eps)
        logits2 = (K2 * proj2.weight).sum(dim=-1)
        weights2 = logits2.softmax(dim=0)
        out2 = (weights2.unsqueeze(-1) * V2).sum(dim=0)
        out2.sum().backward()

        # Gradients must match
        torch.testing.assert_close(proj1.weight.grad, proj2.weight.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(norm1.weight.grad, norm2.weight.grad, atol=1e-6, rtol=1e-5)
        for i in range(5):
            torch.testing.assert_close(
                srcs1[i].grad, srcs2[i].grad, atol=1e-6, rtol=1e-5,
                msg=f"Source {i} gradient mismatch",
            )


def _block_attn_res_loop(sources, proj, norm):
    """Old per-source loop implementation for benchmarking comparison."""
    w = proj.weight
    logits = torch.stack([(norm(v) * w).sum(dim=-1) for v in sources])
    weights = logits.softmax(dim=0)
    V = torch.stack(sources)
    return (weights.unsqueeze(-1) * V).sum(dim=0)


class TestBlockAttnResTPS(unittest.TestCase):
    """Benchmark comparing batched vs per-source loop implementations."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for TPS benchmark")
    def test_batched_vs_loop_benchmark(self):
        """Compare full block_attn_res (batched) vs per-source loop at multiple scales.

        This is an informational benchmark — it prints speedup ratios at
        different model scales to guide TPS optimization decisions. The real
        TPS improvement is measured by running the full model (Task 15.4/15.5).
        """
        configs = [
            # (label, B, T, D, N)
            ("debugmodel (D=256)", 16, 2048, 256, 9),
            ("1B (D=2048)", 1, 8192, 2048, 9),
            ("8B (D=4096)", 1, 8192, 4096, 9),
        ]

        print("\n  Block AttnRes: batched F.rms_norm vs per-source loop")
        print("  " + "-" * 60)

        for label, B, T, D, N in configs:
            proj = nn.Linear(D, 1, bias=False).cuda()
            norm = nn.RMSNorm(D).cuda()
            blocks = [torch.randn(B, T, D, device="cuda") for _ in range(N - 1)]
            partial = torch.randn(B, T, D, device="cuda")
            sources = blocks + [partial]

            # Warmup both paths
            for _ in range(5):
                _ = block_attn_res(blocks, partial, proj, norm)
                _ = _block_attn_res_loop(sources, proj, norm)
            torch.cuda.synchronize()

            n_iters = 50

            # Per-source loop (old)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = _block_attn_res_loop(sources, proj, norm)
            torch.cuda.synchronize()
            loop_time = time.perf_counter() - t0

            # Batched (new — the actual block_attn_res function)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = block_attn_res(blocks, partial, proj, norm)
            torch.cuda.synchronize()
            batch_time = time.perf_counter() - t0

            speedup = loop_time / batch_time
            print(f"  {label:25s}  loop={loop_time:.4f}s  batch={batch_time:.4f}s  {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
