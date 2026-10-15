"""Unit test suite verifying logic controls across the data schemas."""
import unittest
from pathlib import Path
from models import DroneImageMetadata
from utils import convert_to_decimal

class TestSplatPipeline(unittest.TestCase):

    def test_metadata_instantiation(self) -> None:
        """Verifies data model encapsulation and structural float access properties."""
        dummy_path = Path("/mock/image.jpg")
        meta = DroneImageMetadata(file_path=dummy_path, coordinates=(37.1234, -120.5678))
        self.assertEqual(meta.latitude, 37.1234)
        self.assertEqual(meta.longitude, -120.5678)

    def test_coordinate_conversion(self) -> None:
        """Verifies DMS array parsing behaves accurately against reference directions."""
        mock_dms = (37, 30, 0) # 37 degrees, 30 minutes, 0 seconds = 37.5
        decimal_val = convert_to_decimal(mock_dms, 'N')
        self.assertAlmostEqual(decimal_val, 37.5)
        
        decimal_west = convert_to_decimal(mock_dms, 'W')
        self.assertAlmostEqual(decimal_west, -37.5)

if __name__ == "__main__":
    unittest.main()
