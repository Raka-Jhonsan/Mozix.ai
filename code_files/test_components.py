"""CPU smoke tests for the app-aligned GRPO components.

Run from this directory:
    python -m unittest -v test_components.py
"""

import unittest

import numpy as np
import torch
from PIL import Image

from components import (
    FEATURE_DIM,
    FrameSelectorMLP,
    GRPOFrameSelector,
    compute_mcq_reward,
    compute_motion_scores,
    grpo_loss,
    select_diverse_indices,
)


class ComponentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        np.random.seed(7)

    def test_selector_uses_app_feature_dimension(self):
        model = FrameSelectorMLP()
        self.assertEqual(model.input_dim, FEATURE_DIM)
        scores = model(torch.randn(6, FEATURE_DIM))
        self.assertEqual(tuple(scores.shape), (6,))

    def test_candidate_group_has_unique_fixed_size_subsets(self):
        selector = GRPOFrameSelector(input_dim=FEATURE_DIM, device="cpu")
        probs = torch.linspace(0.1, 0.9, 32)
        candidates = selector.generate_candidates(probs, k=8, num_candidates=8)
        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(len(candidates), len({tuple(x) for x in candidates}))
        self.assertTrue(all(len(x) == 8 for x in candidates))
        self.assertTrue(all(0 <= i < 32 for x in candidates for i in x))

    def test_app_temporal_diversity_rule(self):
        scores = np.linspace(1.0, 0.0, 20, dtype=np.float32)
        selected = select_diverse_indices(scores, k=4)
        self.assertEqual(selected, [0, 2, 4, 6])

    def test_grpo_loss_backpropagates_and_handles_equal_rewards(self):
        logits = torch.randn(16, requires_grad=True)
        probs = torch.sigmoid(logits)
        candidates = [
            list(range(0, 8)),
            list(range(8, 16)),
            [0, 2, 4, 6, 8, 10, 12, 14],
        ]
        loss = grpo_loss(probs, candidates, [1.0, 0.0, 1.0], T=16)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

        equal_loss = grpo_loss(probs, candidates, [1.0, 1.0, 1.0], T=16)
        self.assertTrue(torch.isfinite(equal_loss))
        self.assertAlmostEqual(float(equal_loss.detach()), 0.0, places=7)

    def test_mcq_reward_resolves_labels_and_avoids_substring_false_positive(self):
        self.assertEqual(compute_mcq_reward("The answer is dog.", "B", ["cat", "dog"]), 1.0)
        self.assertEqual(compute_mcq_reward("A dog is running.", "A"), 0.0)
        self.assertEqual(compute_mcq_reward("red carpet", "car"), 0.0)
        self.assertEqual(compute_mcq_reward("red car", "red car"), 1.0)

    def test_motion_scores_are_normalized(self):
        frames = [
            Image.new("RGB", (128, 128), (0, 0, 0)),
            Image.new("RGB", (128, 128), (32, 32, 32)),
            Image.new("RGB", (128, 128), (32, 32, 32)),
        ]
        scores = compute_motion_scores(frames)
        self.assertEqual(scores.shape, (3,))
        self.assertEqual(float(scores[0]), 0.0)
        self.assertAlmostEqual(float(scores.max()), 1.0, places=6)
        self.assertTrue(np.isfinite(scores).all())


if __name__ == "__main__":
    unittest.main()
