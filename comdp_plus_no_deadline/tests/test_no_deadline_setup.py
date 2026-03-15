import unittest

from comdp_plus_no_deadline.domains import NasaRoverNoDeadline
from comdp_plus_no_deadline.engines import MDP
from comdp_plus_no_deadline.run_no_deadline import build_regular_problem
from unified_planning.engines.utils import create_init_stn


class TestNoDeadlineSetup(unittest.TestCase):
    def test_domain_has_no_deadline(self):
        model = NasaRoverNoDeadline(
            kind="regular",
            deadline=None,
            object_amount=1,
            garbage_amount=0,
        )
        self.assertIsNone(model.problem.deadline)

    def test_mdp_reward_when_deadline_missing(self):
        converted_problem = build_regular_problem("stuck_car_1o", 1, 0)
        mdp = MDP(converted_problem, discount_factor=0.95, reward_mode="deadline")
        reward = mdp.terminal_reward(True, mdp.initial_state())
        self.assertEqual(reward, 1)

    def test_initial_stn_without_deadline(self):
        converted_problem = build_regular_problem("stuck_car_1o", 1, 0)
        mdp = MDP(converted_problem, discount_factor=0.95, reward_mode="terminal")
        stn = create_init_stn(mdp)
        self.assertTrue(stn.is_consistent())


if __name__ == "__main__":
    unittest.main()

