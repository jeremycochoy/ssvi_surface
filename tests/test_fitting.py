import unittest
import numpy as np
import numpy.testing as npt
from ssvi_surface import SSVIModel, forward_bs_price


class TestFitting(unittest.TestCase):
    """Test fitting method properties."""
    
    def setUp(self):
        self.model = SSVIModel(lr=1e-3, outside_spread_penalty=0.0)
        self.S0 = 100.0
        self.r = 0.025
        self.q = 0.0
        
        # Create synthetic option data
        self.T_array = np.array([0.25, 0.5, 1.0])
        self.strikes_list = [
            np.array([80, 90, 100, 110, 120]),
            np.array([80, 90, 100, 110, 120]),
            np.array([80, 90, 100, 110, 120])
        ]
        
        # Generate synthetic bid/ask prices using a simple model
        self.bids_list = []
        self.asks_list = []
        self.option_types_list = []
        
        for T, strikes in zip(self.T_array, self.strikes_list):
            F_T = self.S0 * np.exp((self.r - self.q) * T)
            k = np.log(strikes / F_T)
            # Use a simple IV model
            iv = 0.2 + 0.1 * k**2
            iv = np.maximum(iv, 0.05)
            
            # Generate call prices (use calls for simplicity)
            call_prices = forward_bs_price(self.S0, strikes, T, iv, self.r, self.q, True)
            
            # Create bid/ask with spread
            spread = 0.02
            bids = call_prices - spread / 2
            asks = call_prices + spread / 2
            option_types = np.array(['call'] * len(strikes))
            
            self.bids_list.append(bids)
            self.asks_list.append(asks)
            self.option_types_list.append(option_types)
            
    def test_parameter_bounds(self):
        """Property: Fitted rho, eta, gamma should be within specified bounds."""
        result = self.model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        self.assertGreaterEqual(self.model.rho, -0.9)
        self.assertLessEqual(self.model.rho, 0.9)
        self.assertGreaterEqual(self.model.eta, 1e-3)
        self.assertLessEqual(self.model.eta, 3.0)
        self.assertGreaterEqual(self.model.gamma, 1e-4)
        self.assertLessEqual(self.model.gamma, 0.99)
        
    def test_arbitrage_free_constraint(self):
        """Property: Should satisfy 4 - eta * (1 + |rho|) > 0 after fitting."""
        result = self.model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        constraint_value = 4 - self.model.eta * (1 + np.abs(self.model.rho))
        self.assertGreater(constraint_value, 0, "Arbitrage-free constraint should be satisfied")
        
    def test_theta_monotonicity(self):
        """Property: theta_t should be monotonic (non-decreasing) if constraint is active."""
        result = self.model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        if len(self.model.theta_t) > 1:
            theta_diff = np.diff(self.model.theta_t)
            self.assertTrue(np.all(theta_diff >= -1e-6), "theta_t should be non-decreasing")
            
    def test_parameter_persistence(self):
        """Property: Fitted parameters should be stored correctly."""
        result = self.model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        self.assertIsNotNone(self.model.rho)
        self.assertIsNotNone(self.model.eta)
        self.assertIsNotNone(self.model.gamma)
        self.assertIsNotNone(self.model.theta_t)
        self.assertIsNotNone(self.model.T_fitted)
        self.assertEqual(len(self.model.theta_t), len(self.T_array))
        
    def test_consistency_after_fitting(self):
        """Property: After fitting, predict() should produce reasonable IVs."""
        result = self.model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        # Test prediction on input strikes
        for T, strikes in zip(self.T_array, self.strikes_list):
            F_T = self.S0 * np.exp((self.r - self.q) * T)
            k = np.log(strikes / F_T)
            iv = self.model.predict(k, T)
            
            # IV should be positive and finite
            self.assertTrue(np.all(iv > 0))
            self.assertTrue(np.all(np.isfinite(iv)))
            self.assertTrue(np.all(iv < 5.0))  # Reasonable upper bound
            
    def test_loss_behavior(self):
        """Property: Objective function should be finite and non-negative after fitting."""
        result = self.model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        
        self.assertTrue(np.isfinite(result.fun))
        self.assertGreaterEqual(result.fun, 0)
        
    def test_rate_estimation(self):
        """Property: If r or q are None, should estimate reasonable values."""
        model = SSVIModel()
        result = model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, None, None
        )
        
        # Estimated rates should be within reasonable bounds
        self.assertGreaterEqual(model.r_fitted, -0.1)
        self.assertLessEqual(model.r_fitted, 0.1)
        self.assertGreaterEqual(model.q_fitted, -0.1)
        self.assertLessEqual(model.q_fitted, 0.1)
        
    def test_rate_usage(self):
        """Property: If r or q are provided, should use them."""
        model = SSVIModel()
        r_provided = 0.03
        q_provided = 0.01
        result = model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, r_provided, q_provided
        )
        
        self.assertEqual(model.r_fitted, r_provided)
        self.assertEqual(model.q_fitted, q_provided)
        
    def test_robustness(self):
        """Property: Small changes in input data should not cause dramatic parameter changes."""
        np.random.seed(42)  # For reproducibility
        result1 = self.model.fit(
            self.T_array, self.strikes_list, self.bids_list, 
            self.asks_list, self.option_types_list, self.S0, self.r, self.q
        )
        params1 = np.array([self.model.rho, self.model.eta, self.model.gamma])
        
        # Perturb data slightly (very small perturbation)
        bids_list_perturbed = [bids + np.random.normal(0, 0.001, len(bids)) 
                               for bids in self.bids_list]
        asks_list_perturbed = [asks + np.random.normal(0, 0.001, len(asks)) 
                               for asks in self.asks_list]
        
        model2 = SSVIModel(lr=1e-3, outside_spread_penalty=0.0)
        result2 = model2.fit(
            self.T_array, self.strikes_list, bids_list_perturbed, 
            asks_list_perturbed, self.option_types_list, self.S0, self.r, self.q
        )
        params2 = np.array([model2.rho, model2.eta, model2.gamma])
        
        # Parameters should not change dramatically (allow for optimization variability)
        param_change = np.abs(params2 - params1)
        # Allow larger tolerance since optimization can be sensitive
        self.assertTrue(np.all(param_change < 1.0), 
                       f"Parameters should be robust to small data changes. Changes: {param_change}")


if __name__ == '__main__':
    unittest.main()

