import unittest
import sys, os

testdir = os.path.dirname(__file__)
srcdir = '../../../../src/main/python/daggit'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from daggit.core.base.parser import get_nodes


class UnitTests(unittest.TestCase):
    def test_get_nodes(self):
        yaml_location = 'test_graph.yaml'
        nodes_bag = get_nodes(yaml_location)
        self.assertEqual(type(nodes_bag), dict)
        self.assertGreater(len(nodes_bag.keys()), 0)
