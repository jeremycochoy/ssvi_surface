import unittest
import numpy as np
import numpy.testing as npt
from ssvi_surface import SSVIModel


class TestHelperMethods(unittest.TestCase):
    """Test helper methods: _fill_spreads, _compute_bid_residuals, _compute_ask_residuals."""
    
    def setUp(self):
        self.model = SSVIModel()
        
    def test_fill_spreads_complete_data(self):
        """Test with complete bid/ask data (no filling needed)."""
        k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        bids = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        asks = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
        
        bids_filled, asks_filled = self.model._fill_spreads(k, bids, asks)
        
        npt.assert_array_equal(bids_filled, bids)
        npt.assert_array_equal(asks_filled, asks)
        
    def test_fill_spreads_missing_bids(self):
        """Test with missing bids (should fill with minimum bid value)."""
        k = np.array([-0.1, 0.0, 0.1])
        bids = np.array([np.nan, 0.7, np.nan])
        asks = np.array([0.7, 0.8, 0.9])
        
        bids_filled, asks_filled = self.model._fill_spreads(k, bids, asks)
        
        # Should return arrays of same length
        self.assertEqual(len(bids_filled), len(k))
        self.assertEqual(len(asks_filled), len(k))
        
        # Missing bids should be filled (no NaN where there are valid asks)
        # The middle element has both bid and ask, so it should remain unchanged
        self.assertFalse(np.isnan(bids_filled[1]), "Valid bid should remain")
        # The first and third should be filled if possible
        # Since there's a valid pair at index 1, they should be filled
        self.assertFalse(np.isnan(bids_filled[0]), "Missing bid should be filled")
        self.assertFalse(np.isnan(bids_filled[2]), "Missing bid should be filled")
        
        self.assertFalse(np.any(np.isnan(bids_filled)))
        self.assertFalse(np.any(np.isnan(asks_filled)))
        
    def test_fill_spreads_missing_asks(self):
        """Test with missing asks (should fill with maximum ask value)."""
        k = np.array([-0.1, 0.0, 0.1])
        bids = np.array([0.5, 0.6, 0.7])
        asks = np.array([np.nan, 0.7, np.nan])
        
        bids_filled, asks_filled = self.model._fill_spreads(k, bids, asks)
        
        # Should return arrays of same length
        self.assertEqual(len(bids_filled), len(k))
        self.assertEqual(len(asks_filled), len(k))
        
        # Missing asks should be filled (no NaN where there are valid bids)
        # The middle element has both bid and ask, so it should remain unchanged
        self.assertFalse(np.isnan(asks_filled[1]), "Valid ask should remain")
        # The first and third should be filled if possible
        # Since there's a valid pair at index 1, they should be filled
        self.assertFalse(np.isnan(asks_filled[0]), "Missing ask should be filled")
        self.assertFalse(np.isnan(asks_filled[2]), "Missing ask should be filled")
        
        self.assertFalse(np.any(np.isnan(asks_filled)))
        self.assertFalse(np.any(np.isnan(bids_filled)))
            
    def test_compute_bid_residuals(self):
        """Test _compute_bid_residuals with valid data."""
        bids = np.array([0.5, 0.6, 0.7])
        model_prices = np.array([0.55, 0.65, 0.75])
        spreads = np.array([0.1, 0.1, 0.1])
        
        residuals, weights = self.model._compute_bid_residuals(bids, model_prices, spreads)
        
        # Residuals should be bids - model_prices
        expected_residuals = bids - model_prices
        npt.assert_array_equal(residuals, expected_residuals)
        
        # Weights should be 1/spread
        expected_weights = 1.0 / spreads
        npt.assert_array_equal(weights, expected_weights)
        
    def test_compute_ask_residuals(self):
        """Test _compute_ask_residuals with valid data."""
        asks = np.array([0.6, 0.7, 0.8])
        model_prices = np.array([0.55, 0.65, 0.75])
        spreads = np.array([0.1, 0.1, 0.1])
        
        residuals, weights = self.model._compute_ask_residuals(asks, model_prices, spreads)
        
        # Residuals should be asks - model_prices
        expected_residuals = asks - model_prices
        npt.assert_array_equal(residuals, expected_residuals)
        
        # Weights should be 1/spread
        expected_weights = 1.0 / spreads
        npt.assert_array_equal(weights, expected_weights)
        
    def test_compute_residuals_with_nan(self):
        """Test residual computation with NaN values."""
        bids = np.array([0.5, np.nan, 0.7])
        model_prices = np.array([0.55, 0.65, 0.75])
        spreads = np.array([0.1, np.nan, 0.1])
        
        residuals, weights = self.model._compute_bid_residuals(bids, model_prices, spreads)
        
        # Should only return valid entries
        self.assertEqual(len(residuals), 2)  # Only first and third are valid
        self.assertEqual(len(weights), 2)
    
    def test_fill_spreads_mixed_missing(self):
        """Test with mixed missing bids and asks."""
        k = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
        bids = np.array([0.5, np.nan, 0.7, np.nan, 0.9])
        asks = np.array([np.nan, 0.7, 0.8, np.nan, 1.0])
        
        bids_filled, asks_filled = self.model._fill_spreads(k, bids, asks)
        
        self.assertFalse(np.any(np.isnan(bids_filled)))
        self.assertFalse(np.any(np.isnan(asks_filled)))


if __name__ == '__main__':
    unittest.main()

