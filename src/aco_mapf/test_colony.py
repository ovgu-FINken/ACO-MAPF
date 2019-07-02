from unittest import TestCase

from src.aco_mapf.AcoAgent import Colony
import numpy as np


class TestColony(TestCase):
    def test_vaporize(self):
        c = Colony(pheromones=np.ones((3,3)))
        c.ants = [1,2]
        c.vaporize(evaporation_method="normalize")
        self.assertAlmostEqual(1 / 9, c.pheromones[0,0])
        c.vaporize(evaporation_method="normalize")
        self.assertAlmostEqual(1 / 9, c.pheromones[0,0])
        c.pheromones = np.ones((3,3))
        self.assertEqual(1, c.pheromones[0,0])
        for _ in c.ants:
            c.vaporize(evaporation_method="default_aco", evaporation_rate=0.7)
        self.assertAlmostEqual(c.pheromones[0,0], 0.7)

