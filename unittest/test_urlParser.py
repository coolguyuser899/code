import unittest
from urlParser import urlParser

class BasicProtocalParsing(unittest.TestCase):

    def test_protocol(self, ):
        purl = urlParser('http://localhost')
        self.assertEqual(purl.protocol, 'http')

