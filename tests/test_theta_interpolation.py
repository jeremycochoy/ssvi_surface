import unittest
import numpy as np
import numpy.testing as npt
from ssvi_surface import SSVIModel


class TestThetaInterpolation(unittest.TestCase):
    """Test theta interpolation properties."""
    
    def setUp(self):
        self.model = SSVIModel()
        # Fit model with synthetic data
        self.model.T_fitted = np.array([0.1, 0.25, 0.5, 1.0])
        self.model.theta_t = np.array([0.01, 0.02, 0.03, 0.04])
        
    def test_error_before_fitting(self):
        """Property: Should raise ValueError before fitting."""
        model = SSVIModel()
        with self.assertRaises(ValueError):
            model.theta_interp(0.5)
            
    def test_exact_match_at_fitted_points(self):
        """Property: Interpolated values at fitted T points should match theta_t exactly."""
        for T, theta_expected in zip(self.model.T_fitted, self.model.theta_t):
            theta_interp = self.model.theta_interp(T)
            npt.assert_allclose(theta_interp, theta_expected, rtol=1e-10, atol=1e-10)
            
    def test_monotonicity_preservation(self):
        """Property: If theta_t is monotonic, interpolation should preserve monotonicity."""
        # theta_t is monotonic (increasing)
        T_test = np.linspace(0.1, 1.0, 20)
        theta_interp = self.model.theta_interp(T_test)
        
        # Check that interpolated values are non-decreasing
        diff = np.diff(theta_interp)
        self.assertTrue(np.all(diff >= -1e-10), "Interpolated theta should be non-decreasing")
        
    def test_shape_preservation(self):
        """Property: Output shape should match input T shape."""
        # Scalar input
        T_scalar = 0.5
        theta_scalar = self.model.theta_interp(T_scalar)
        self.assertEqual(theta_scalar.shape, ())
        
        # Array input
        T_array = np.array([0.2, 0.4, 0.6])
        theta_array = self.model.theta_interp(T_array)
        self.assertEqual(theta_array.shape, (3,))
        
    def test_continuity(self):
        """Property: Interpolated values should be continuous."""
        T_base = np.linspace(0.15, 0.95, 50)
        theta_base = self.model.theta_interp(T_base)
        
        # Small perturbation
        T_perturbed = T_base + 1e-6
        theta_perturbed = self.model.theta_interp(T_perturbed)
        
        # Changes should be small
        max_change = np.max(np.abs(theta_perturbed - theta_base))
        self.assertLess(max_change, 1e-4, "Interpolation should be continuous")
        
    def test_interpolation_bounds(self):
        """Property: Interpolated values should be within reasonable range."""
        T_test = np.linspace(0.05, 1.5, 100)  # Include extrapolation
        theta_interp = self.model.theta_interp(T_test)
        
        # Should be finite and positive
        self.assertTrue(np.all(np.isfinite(theta_interp)))
        self.assertTrue(np.all(theta_interp > 0))
        
    def test_extrapolation(self):
        """Property: Should handle extrapolation beyond fitted range."""
        # Extrapolate before first point
        T_before = 0.05
        theta_before = self.model.theta_interp(T_before)
        self.assertTrue(np.isfinite(theta_before))
        
        # Extrapolate after last point
        T_after = 2.0
        theta_after = self.model.theta_interp(T_after)
        self.assertTrue(np.isfinite(theta_after))


if __name__ == '__main__':
    unittest.main()

