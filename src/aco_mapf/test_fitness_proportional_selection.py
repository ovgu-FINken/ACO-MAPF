import unittest

from src.aco_mapf.AcoAgent import fitness_proportional_selection


class TestFitness_proportional_selection(unittest.TestCase):

    def test_zero_vs_nonzero(self):
        probs = {'never': 0.0, 'always': 1.0}
        for _ in range(10):
            self.assertEqual(fitness_proportional_selection(probs), 'always')

    def test_different_probs(self):
        probs = {'first':0.1, 'second': 0.5, 'third': 0.4}
        self.assertEqual(fitness_proportional_selection(probs, 0.05), 'first')
        self.assertEqual(fitness_proportional_selection(probs, 0.15), 'second')
        self.assertEqual(fitness_proportional_selection(probs, 0.55), 'second')
        self.assertEqual(fitness_proportional_selection(probs, 0.99), 'third')

if __name__ == '__main__':
    unittest.main()