import unittest
import numpy as np
import pandas as pd
from ssvi_surface import SSVIModel, forward_bs_price


class TestIntegration(unittest.TestCase):
    """Integration tests for full workflow."""
    
    def setUp(self):
        self.S0 = 100.0
        self.r = 0.025
        self.q = 0.0
        
        # Create realistic option chain data
        self.T_array = np.array([0.25, 0.5, 1.0])
        self.strikes_list = [
            np.array([80, 90, 100, 110, 120]),
            np.array([80, 90, 100, 110, 120]),
            np.array([80, 90, 100, 110, 120])
        ]
        
        # Generate synthetic bid/ask prices
        self.bids_list = []
        self.asks_list = []
        self.option_types_list = []
        
        for T, strikes in zip(self.T_array, self.strikes_list):
            F_T = self.S0 * np.exp((self.r - self.q) * T)
            k = np.log(strikes / F_T)
            iv = 0.2 + 0.1 * k**2
            iv = np.maximum(iv, 0.05)
            
            call_prices = forward_bs_price(self.S0, strikes, T, iv, self.r, self.q, True)
            
            spread = 0.02
            bids = call_prices - spread / 2
            asks = call_prices + spread / 2
            option_types = np.array(['call'] * len(strikes))
            
            self.bids_list.append(bids)
            self.asks_list.append(asks)
            self.option_types_list.append(option_types)
            
        # Create option DataFrame
        rows = []
        for i, (T, strikes) in enumerate(zip(self.T_array, self.strikes_list)):
            for j, strike in enumerate(strikes):
                rows.append({
                    'expiry': f'2024-{i+1:02d}-01',
                    'strike': strike,
                    'option_type': 'call',
                    'T': T,
                    'bid_usd': self.bids_list[i][j],
                    'ask_usd': self.asks_list[i][j]
                })
        self.option_df = pd.DataFrame(rows)
        
    def test_full_workflow(self):
        """Property: Full workflow should complete successfully."""
        model = SSVIModel()
        
        # Fit
        result = model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        # Predict
        k = np.linspace(-0.5, 0.5, 10)
        t = 0.5
        iv = model.predict(k, t)
        
        # Check crossings
        crossings_df = model.check_bid_ask_crossings(self.option_df, self.S0, self.r, self.q)
        
        # All should complete without errors
        self.assertIsNotNone(model.rho)
        self.assertTrue(np.all(iv > 0))
        self.assertGreater(len(crossings_df), 0)
        
    def test_parameter_persistence(self):
        """Property: Parameters should persist after fitting and be usable."""
        model = SSVIModel()
        result = model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        # Store parameters
        rho = model.rho
        eta = model.eta
        gamma = model.gamma
        theta_t = model.theta_t.copy()
        
        # Use in prediction
        k = 0.1
        t = 0.5
        iv = model.predict(k, t)
        
        # Parameters should still be the same
        self.assertEqual(model.rho, rho)
        self.assertEqual(model.eta, eta)
        self.assertEqual(model.gamma, gamma)
        np.testing.assert_array_equal(model.theta_t, theta_t)
        
    def test_consistency(self):
        """Property: predict() results should be consistent with fitted model."""
        model = SSVIModel()
        result = model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        # Test prediction on input strikes
        for T, strikes in zip(self.T_array, self.strikes_list):
            F_T = self.S0 * np.exp((self.r - self.q) * T)
            k = np.log(strikes / F_T)
            iv = model.predict(k, T)
            
            # IV should be consistent (positive, finite, reasonable range)
            self.assertTrue(np.all(iv > 0))
            self.assertTrue(np.all(np.isfinite(iv)))
            self.assertTrue(np.all(iv < 2.0))  # Reasonable upper bound for IV


if __name__ == '__main__':
    unittest.main()

