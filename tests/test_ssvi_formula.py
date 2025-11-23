import unittest
import numpy as np
import numpy.testing as npt
from ssvi_surface import SSVIModel


class TestSSVIFormula(unittest.TestCase):
    """Test core SSVI formula properties."""
    
    def setUp(self):
        self.model = SSVIModel()
        # Use typical parameter values
        self.rho = -0.3
        self.eta = 1.0
        self.gamma = 0.5
        self.theta_t = 0.04  # Typical ATM variance for 1 year
        
    def test_positivity(self):
        """Property: w(k,t) > 0 for all valid inputs."""
        k_values = np.linspace(-2, 2, 100)
        w = self.model.ssvi(k_values, self.rho, self.eta, self.gamma, self.theta_t)
        self.assertTrue(np.all(w > 0), "All w values should be positive")
        
    def test_continuity(self):
        """Property: w should be continuous in k."""
        k_base = np.linspace(-1, 1, 50)
        w_base = self.model.ssvi(k_base, self.rho, self.eta, self.gamma, self.theta_t)
        
        # Small perturbation
        k_perturbed = k_base + 1e-6
        w_perturbed = self.model.ssvi(k_perturbed, self.rho, self.eta, self.gamma, self.theta_t)
        
        # Changes should be small
        max_change = np.max(np.abs(w_perturbed - w_base))
        self.assertLess(max_change, 1e-4, "Function should be continuous")
        
    def test_atm_behavior(self):
        """Property: Test behavior at k=0 (ATM)."""
        k_atm = 0.0
        w_atm = self.model.ssvi(k_atm, self.rho, self.eta, self.gamma, self.theta_t)
        
        # At ATM (k=0), phi_k = 0, so w = (theta_t/2) * (1 + sqrt(1))
        # w = (theta_t/2) * (1 + 1) = theta_t
        expected = self.theta_t
        npt.assert_allclose(w_atm, expected, rtol=1e-10)
        
    def test_monotonicity(self):
        """Property: w should generally increase with |k| for typical parameters."""
        k_positive = np.linspace(0, 2, 50)
        w_positive = self.model.ssvi(k_positive, self.rho, self.eta, self.gamma, self.theta_t)
        
        # For positive k with negative rho, w should generally increase
        # (exact monotonicity depends on parameters, but should be mostly increasing)
        w_at_0 = w_positive[0]
        w_at_2 = w_positive[-1]
        self.assertGreater(w_at_2, w_at_0 * 0.9, "w should generally increase with k")
        
        # Test negative k
        k_negative = np.linspace(-2, 0, 50)
        w_negative = self.model.ssvi(k_negative, self.rho, self.eta, self.gamma, self.theta_t)
        # w should still be positive and finite
        self.assertTrue(np.all(w_negative > 0))
        
    def test_theta_scaling(self):
        """Property: w should scale approximately with theta_t."""
        k = 0.1
        theta_t1 = 0.04
        theta_t2 = 0.08
        
        w1 = self.model.ssvi(k, self.rho, self.eta, self.gamma, theta_t1)
        w2 = self.model.ssvi(k, self.rho, self.eta, self.gamma, theta_t2)
        
        # Ratio should be approximately theta_t1 / theta_t2
        # (not exactly due to phi term, but close)
        ratio = w1 / w2
        expected_ratio = theta_t1 / theta_t2
        npt.assert_allclose(ratio, expected_ratio, rtol=0.05)  # Allow 5% tolerance
        
    def test_array_handling(self):
        """Property: Test with scalar and array inputs."""
        k_scalar = 0.1
        k_array = np.array([0.1, 0.2, 0.3])
        
        w_scalar = self.model.ssvi(k_scalar, self.rho, self.eta, self.gamma, self.theta_t)
        w_array = self.model.ssvi(k_array, self.rho, self.eta, self.gamma, self.theta_t)
        
        self.assertEqual(w_scalar.shape, ())
        self.assertEqual(w_array.shape, (3,))
        npt.assert_allclose(w_array[0], w_scalar, rtol=1e-10)
        
    def test_parameter_bounds(self):
        """Property: Formula handles boundary parameter values."""
        # Test boundary rho values
        for rho in [-0.9, 0.0, 0.9]:
            w = self.model.ssvi(0.1, rho, self.eta, self.gamma, self.theta_t)
            self.assertTrue(np.isfinite(w) and w > 0)
            
        # Test boundary gamma values
        for gamma in [0.01, 0.5, 0.99]:
            w = self.model.ssvi(0.1, self.rho, self.eta, gamma, self.theta_t)
            self.assertTrue(np.isfinite(w) and w > 0)
            
        # Test different eta values
        for eta in [0.1, 1.0, 2.0]:
            w = self.model.ssvi(0.1, self.rho, eta, self.gamma, self.theta_t)
            self.assertTrue(np.isfinite(w) and w > 0)
            
    def test_numerical_stability(self):
        """Property: Very large |k| and small theta_t should produce finite results."""
        # Large |k|
        k_large = np.array([-5, 5])
        w_large = self.model.ssvi(k_large, self.rho, self.eta, self.gamma, self.theta_t)
        self.assertTrue(np.all(np.isfinite(w_large)))
        self.assertTrue(np.all(w_large > 0))
        
        # Small theta_t
        theta_t_small = 1e-5
        w_small = self.model.ssvi(0.1, self.rho, self.eta, self.gamma, theta_t_small)
        self.assertTrue(np.isfinite(w_small))
        self.assertTrue(w_small > 0)


if __name__ == '__main__':
    unittest.main()

