import unittest
import numpy as np
import numpy.testing as npt
from ssvi_surface import SSVIModel


class TestPrediction(unittest.TestCase):
    """Test prediction method properties."""
    
    def setUp(self):
        self.model = SSVIModel()
        # Fit model with synthetic data
        self.model.rho = -0.3
        self.model.eta = 1.0
        self.model.gamma = 0.5
        self.model.T_fitted = np.array([0.1, 0.25, 0.5, 1.0])
        self.model.theta_t = np.array([0.01, 0.02, 0.03, 0.04])
        
    def test_error_before_fitting(self):
        """Property: Should raise ValueError before fitting."""
        model = SSVIModel()
        with self.assertRaises(ValueError):
            model.predict(0.1, 0.5)
            
    def test_positivity(self):
        """Property: IV output should always be positive."""
        k_values = np.linspace(-2, 2, 100)
        t_values = np.array([0.25, 0.5, 1.0])
        
        for t in t_values:
            iv = self.model.predict(k_values, t)
            self.assertTrue(np.all(iv > 0), f"All IV values should be positive for t={t}")
            
    def test_consistency_with_ssvi(self):
        """Property: predict(k, t) should equal sqrt(ssvi(...) / t)."""
        k = 0.1
        t = 0.5
        
        iv_predict = self.model.predict(k, t)
        theta_t_val = self.model.theta_interp(t)
        w_ssvi = self.model.ssvi(k, self.model.rho, self.model.eta, self.model.gamma, theta_t_val)
        iv_expected = np.sqrt(np.maximum(w_ssvi / t, 1e-8))
        
        npt.assert_allclose(iv_predict, iv_expected, rtol=1e-10)
        
    def test_shape_preservation(self):
        """Property: Output shape should match input shape."""
        # Scalar inputs
        iv_scalar = self.model.predict(0.1, 0.5)
        self.assertEqual(iv_scalar.shape, ())
        
        # Array k, scalar t
        k_array = np.array([0.1, 0.2, 0.3])
        iv_array = self.model.predict(k_array, 0.5)
        self.assertEqual(iv_array.shape, (3,))
        
        # Scalar k, array t
        t_array = np.array([0.25, 0.5, 1.0])
        iv_t_array = self.model.predict(0.1, t_array)
        self.assertEqual(iv_t_array.shape, (3,))
        
        # Both arrays (broadcasting)
        iv_both = self.model.predict(k_array, t_array)
        self.assertEqual(iv_both.shape, (3,))
        
    def test_monotonicity(self):
        """Property: For fixed t, IV should generally increase with |k|."""
        t = 0.5
        k_positive = np.linspace(0, 2, 50)
        iv_positive = self.model.predict(k_positive, t)
        
        # IV should generally increase with |k| (volatility smile)
        # Check that IV at larger |k| is generally higher
        iv_at_0 = self.model.predict(0.0, t)
        iv_at_1 = self.model.predict(1.0, t)
        iv_at_2 = self.model.predict(2.0, t)
        
        self.assertGreater(iv_at_1, iv_at_0 * 0.9)  # Allow some tolerance
        self.assertGreater(iv_at_2, iv_at_1 * 0.9)
        
    def test_time_behavior(self):
        """Property: For fixed k, IV should behave reasonably with respect to t."""
        k = 0.1
        t_values = np.array([0.1, 0.25, 0.5, 1.0])
        iv_values = self.model.predict(k, t_values)
        
        # All should be positive and finite
        self.assertTrue(np.all(iv_values > 0))
        self.assertTrue(np.all(np.isfinite(iv_values)))


if __name__ == '__main__':
    unittest.main()

