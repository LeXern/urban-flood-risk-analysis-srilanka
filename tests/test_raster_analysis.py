"""
Tests for raster analysis module.
"""

import numpy as np
import pytest
from src.raster_analysis import (
    create_extreme_rainfall_mask,
    count_extreme_events,
    normalize_array,
    calculate_vulnerability_index
)


class TestExtremeRainfallMask:
    """Tests for extreme rainfall detection."""
    
    def test_basic_threshold(self):
        """Test that threshold masking works correctly."""
        data = np.array([50, 100, 150, 200])
        mask = create_extreme_rainfall_mask(data, threshold=100)
        
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(mask, expected)
    
    def test_all_below_threshold(self):
        """Test when all values are below threshold."""
        data = np.array([10, 20, 30])
        mask = create_extreme_rainfall_mask(data, threshold=100)
        
        assert not mask.any(), "No values should exceed threshold"
    
    def test_all_above_threshold(self):
        """Test when all values are above threshold."""
        data = np.array([150, 200, 250])
        mask = create_extreme_rainfall_mask(data, threshold=100)
        
        assert mask.all(), "All values should exceed threshold"


class TestCountExtremeEvents:
    """Tests for counting extreme events."""
    
    def test_count_along_time_axis(self):
        """Test counting extreme events along time axis."""
        # 3 days, 2x2 grid
        data = np.array([
            [[50, 150], [200, 100]],   # day 1
            [[120, 80], [50, 150]],    # day 2
            [[150, 150], [150, 150]]   # day 3
        ])
        
        counts = count_extreme_events(data, threshold=100)
        
        expected = np.array([[2, 2], [2, 2]])
        np.testing.assert_array_equal(counts, expected)


class TestNormalizeArray:
    """Tests for normalization functions."""
    
    def test_minmax_normalization(self):
        """Test min-max normalization to [0, 1]."""
        data = np.array([0, 50, 100])
        normalized = normalize_array(data, method='minmax')
        
        expected = np.array([0.0, 0.5, 1.0])
        np.testing.assert_array_almost_equal(normalized, expected)
    
    def test_minmax_range(self):
        """Test that normalized values are in [0, 1]."""
        data = np.random.rand(100) * 1000
        normalized = normalize_array(data, method='minmax')
        
        assert normalized.min() >= 0, "Min should be >= 0"
        assert normalized.max() <= 1, "Max should be <= 1"
    
    def test_constant_array(self):
        """Test normalization of constant array (avoid division by zero)."""
        data = np.ones(10) * 50
        normalized = normalize_array(data, method='minmax')
        
        assert np.all(normalized == 0), "Constant array should normalize to zeros"


class TestVulnerabilityIndex:
    """Tests for vulnerability index calculation."""
    
    def test_equal_weights(self):
        """Test vulnerability calculation with equal weights."""
        rainfall = np.array([0.5])
        building = np.array([0.5])
        elevation = np.array([0.5])
        
        vuln = calculate_vulnerability_index(
            rainfall, building, elevation,
            weights=(1/3, 1/3, 1/3)
        )
        
        # elevation inverted: 1 - 0.5 = 0.5
        # all components = 0.5, so result = 0.5
        np.testing.assert_almost_equal(vuln[0], 0.5)
    
    def test_high_vulnerability_case(self):
        """Test high vulnerability scenario."""
        rainfall = np.array([1.0])      # max rainfall
        building = np.array([1.0])      # max density
        elevation = np.array([0.0])     # low elevation (inverts to 1.0)
        
        vuln = calculate_vulnerability_index(
            rainfall, building, elevation,
            weights=(0.4, 0.3, 0.3)
        )
        
        # 0.4*1 + 0.3*1 + 0.3*1 = 1.0
        np.testing.assert_almost_equal(vuln[0], 1.0)
    
    def test_low_vulnerability_case(self):
        """Test low vulnerability scenario."""
        rainfall = np.array([0.0])      # no rainfall
        building = np.array([0.0])      # no buildings
        elevation = np.array([1.0])     # high elevation (inverts to 0)
        
        vuln = calculate_vulnerability_index(
            rainfall, building, elevation,
            weights=(0.4, 0.3, 0.3)
        )
        
        np.testing.assert_almost_equal(vuln[0], 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
