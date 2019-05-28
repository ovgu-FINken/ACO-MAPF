from unittest import TestCase
from src.aco_mapf.GraphWorld import TestProblem


class TestTestProblem(TestCase):
    def test_contrallable_random(self):
        t1 = TestProblem(seed=42).hard_1()
        t2 = TestProblem(seed=42).hard_1()
        t3 = TestProblem(seed=23).hard_1()

        self.assertTrue((t1.adjacency == t2.adjacency).all)
        self.assertTrue((t1.adjacency != t3.adjacency).any)
