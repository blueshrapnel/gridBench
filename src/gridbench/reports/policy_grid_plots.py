"""Policy-grid demo: DI-optimal policy quiver over a per-state decision-info heatmap.

Ported from gridFour experiments/policy-grid-plots/create_plots.py.  Dead imports
(RelevantInformation, information_theory, ValueIteration, duplicate Policy) were
dropped and the module-level body wrapped in main() so importing this module has
no side effects.
"""
import matplotlib.pyplot as plt
import numpy as np

from gridcore.envs import GridRoom
from gridcore.info.decision_information import DecisionInformation
from gridcore.planning.policy import Policy
from gridcore.planning.state_distribution import LiveStateDistribution
import gridvis.display as di


def main():
    shape = (5, 5)
    neighbourhood = 'man'
    goal_desc = 'centre'
    goal = int(np.prod(shape) / 2)
    options = {'shape': shape, 'goals': [goal], 'manhattan': True, 'determinism': 1}
    theta = 1e-5
    beta = 0.1
    env = GridRoom(options)
    state_dist = LiveStateDistribution(env)

    exp_desc = '{}-{}-{:.3}-{:.3}-det-{}-goal-{:.4}'.format(
        options['shape'][0], options['shape'][1], neighbourhood,
        state_dist.name.lower(), env.determinism, goal_desc)
    print(exp_desc)

    ig = DecisionInformation(env, state_dist, theta)
    pi_opt, Z, F = ig.get_opt_policy_Z_free_vector(beta)
    pi = Policy(env, pi_opt)
    I = ig.get_decision_information_given_policy(pi_opt)
    V, Q = pi.get_V_Q()
    print('beta: {}'.format(beta))
    print('state dist: {}'.format(ig.state_dist.name))
    p_s = ig.state_dist.set_ps(pi_opt)
    di.plot_quiver_heats_val(env, p_s, I, pi.get_ndarray(), label=True, clim=(0, 1))
    plt.show()


if __name__ == "__main__":
    main()
