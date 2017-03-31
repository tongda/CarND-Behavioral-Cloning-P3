from unittest import TestCase
from data import read_data

class DataLoaderTestCase(TestCase):
    def test_read_data(self):
        images, measurements = read_data('./data')
        self.assertEqual(images.shape, (5166, 160, 320, 3))