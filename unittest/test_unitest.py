import unittest


"""
to run test discovery in command
python -m unittest discover <folder>, with or without the ‘-v’,
"""
def multiply(a, b):
    """
    >>> multiply(4, 3)
    12
    >>> multiply('a', 3)
    'aaa'
    """
    return a * b

class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_numbers_3_4(self):
        self.assertEqual( multiply(3,4), 12)

    def test_strings_a_3(self):
        self.assertEqual( multiply('a', 3), 'aaa')

if __name__ == '__main__':
    unittest.main()
