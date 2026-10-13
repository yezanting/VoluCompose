"""Small mathematical/output-contract checks, not benchmark reproduction."""
import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_GEOOPT = importlib.util.find_spec("geoopt") is not None


@unittest.skipUnless(HAS_TORCH and HAS_GEOOPT, "requires torch and geoopt")
class CompressionContractTests(unittest.TestCase):
    def test_entropy_capacity_limits_for_uniform_and_concentrated_attention(self):
        import torch
        from medcompose.medpruner_pp.utils import cal_rate
        self.assertAlmostEqual(cal_rate(torch.ones(2, 8)).item(), 0.3, places=6)
        peaked = torch.zeros(2, 8)
        peaked[:, 0] = 1
        self.assertAlmostEqual(cal_rate(peaked).item(), 0.0, places=6)

    def test_grouping_preserves_order_and_averages_corresponding_positions(self):
        import torch
        from medcompose.medpruner_pp.utils import frame_group_fusion
        a = torch.full((4, 3), 0.01)
        b = torch.full((4, 3), 0.2)
        grouped, ids = frame_group_fusion(torch.stack([a, a, b]), uniqueness=0.005)
        self.assertEqual(ids.tolist(), [0, 2])
        torch.testing.assert_close(grouped, torch.stack([a, b]))

    def test_allocated_capacity_and_output_mask_obey_requested_budget(self):
        import torch
        from medcompose.medpruner_pp.utils import token_allocation, spatial_dynamic_compression2
        torch.manual_seed(42)
        features = torch.randn(3, 16, 8) * 0.01
        capacities = token_allocation(features, max_tokens=12, max_token_per_item=16)
        self.assertLessEqual(int(capacities.sum()), 12)
        self.assertTrue(bool((capacities >= 0).all()))
        output, mask = spatial_dynamic_compression2(
            features.clone(), tokens_per_frame=capacities,
            selected_frame_ids=torch.tensor([0, 2, 4]), original_num_frames=5,
            return_token_mask=True,
        )
        self.assertEqual(output.shape[0], int(capacities.sum()))
        self.assertEqual(mask.shape, (5 * 16,))
        self.assertEqual(int(mask.sum()), output.shape[0])
        self.assertEqual(int(mask.view(5, 16)[[1, 3]].sum()), 0)
        self.assertTrue(bool(torch.isfinite(output).all()))


if __name__ == "__main__":
    unittest.main()
