"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for servers or notebooks



class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self):
        """ """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#

class Visualization:
    """
    Contains methods to visualize the performance of bandit algorithms.
    """

    def plot1(self, eg, ts):
        plt.figure(figsize=(12, 5))
        plt.plot(eg.rewards, label='Epsilon-Greedy')
        plt.plot(ts.rewards, label='Thompson Sampling')
        plt.title('Reward Per Trial (Linear Scale)')
        plt.xlabel('Trial')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('img/rewards_linear.png')
        plt.show()

        plt.figure(figsize=(12, 5))
        plt.plot(np.log(np.clip(eg.rewards, a_min=1e-3, a_max=None)), label='Epsilon-Greedy')
        plt.plot(np.log(np.clip(ts.rewards, a_min=1e-3, a_max=None)), label='Thompson Sampling')
        plt.title('Log Reward Per Trial')
        plt.xlabel('Trial')
        plt.ylabel('Log Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('img/rewards_log.png')
        plt.show()

    def plot2(self, eg, ts, p):
        plt.figure(figsize=(12, 5))
        plt.plot(np.cumsum(eg.rewards), label='Epsilon-Greedy')
        plt.plot(np.cumsum(ts.rewards), label='Thompson Sampling')
        plt.title('Cumulative Rewards')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid(True)
        plt.savefig('img/cumulative_rewards.png')
        plt.show()

        optimal = np.max(p)
        eg_regret = np.cumsum(optimal - np.array(eg.rewards))
        ts_regret = np.cumsum(optimal - np.array(ts.rewards))

        plt.figure(figsize=(12, 5))
        plt.plot(eg_regret, label='Epsilon-Greedy Regret')
        plt.plot(ts_regret, label='Thompson Sampling Regret')
        plt.title('Cumulative Regret')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.grid(True)
        plt.savefig('img/cumulative_regret.png')
        plt.show()

    def plot_distributions(self, p, eg_choices, ts_choices):
        num_trials_list = [5, 100, 500, 2000, 5000, 10000, 15000, 20000]
        for n in num_trials_list:
            if len(eg_choices) < n or len(ts_choices) < n:
                continue
            eg_counts = np.bincount(eg_choices[:n], minlength=len(p))
            ts_counts = np.bincount(ts_choices[:n], minlength=len(p))
            x = np.linspace(min(p) - 2, max(p) + 2, 1000)
            plt.figure(figsize=(8, 4))
            for arm in range(len(p)):
                mu = p[arm]
                sigma_eg = 1 / np.sqrt(max(eg_counts[arm], 1))
                sigma_ts = 1 / np.sqrt(max(ts_counts[arm], 1))
                y_eg = 1 / (sigma_eg * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma_eg ** 2))
                y_ts = 1 / (sigma_ts * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma_ts ** 2))
                plt.plot(x, y_eg, label=f"EpsilonGreedy Arm {arm} (Trials: {eg_counts[arm]})", linestyle='--')
                plt.plot(x, y_ts, label=f"ThompsonSampling Arm {arm} (Trials: {ts_counts[arm]})")
            plt.title(f"Bandit distributions after {n} trials")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"img/bandit_dist_{n}_trials.png")
            plt.close()



class EpsilonGreedy(Bandit):
    """Implements the Epsilon-Greedy strategy with 1/t decay."""

    def __init__(self, p):
        """
        Initialize the EpsilonGreedy bandit.

        Args:
            p (list): True reward values for each arm.
        """
        self.p = p
        self.k = len(p)
        self.estimates = np.zeros(self.k)
        self.counts = np.zeros(self.k)
        self.total_reward = 0
        self.rewards = []
        self.choices = []
        self.eps = 1.0

    def __repr__(self):
        """String representation of current estimates."""
        return f"EpsilonGreedy(estimates={self.estimates})"

    def pull(self):
        """Choose an arm to pull using epsilon-greedy logic."""
        if np.random.random() < self.eps:
            return np.random.randint(self.k)
        return np.argmax(self.estimates)

    def update(self, chosen_arm, reward):
        """Update estimated value of the chosen arm.

        Args:
          chosen_arm(int): Index of the selected arm.
          reward(float): Observed reward.

        Returns:

        """
        self.counts[chosen_arm] += 1
        self.estimates[chosen_arm] += (1 / self.counts[chosen_arm]) * (reward - self.estimates[chosen_arm])
        self.total_reward += reward

    def experiment(self, num_trials=20000):
        """Run the epsilon-greedy experiment.

        Args:
          num_trials(int, optional): Number of trials to run. (Default value = 20000)

        Returns:

        """
        for t in range(1, num_trials + 1):
            self.eps = 1 / t
            chosen_arm = self.pull()
            reward = np.random.normal(self.p[chosen_arm], 1)
            self.update(chosen_arm, reward)
            self.rewards.append(reward)
            self.choices.append((chosen_arm, reward))

    def report(self):
        """Generate report: save rewards, print average reward and regret."""
        df = pd.DataFrame([(arm, r, 'EpsilonGreedy') for arm, r in self.choices],
                          columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('data/rewards.csv', mode='a', index=False)
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"EpsilonGreedy Avg Reward: {avg_reward:.4f}")
        logger.info(f"EpsilonGreedy Cumulative Regret: {regret:.4f}")


class ThompsonSampling(Bandit):
    """Implements Thompson Sampling using Beta priors."""

    def __init__(self, p):
        """
        Initialize the ThompsonSampling bandit.

        Args:
            p (list): True reward values for each arm.
        """
        self.p = p
        self.k = len(p)
        self.successes = np.ones(self.k)
        self.failures = np.ones(self.k)
        self.rewards = []
        self.choices = []

    def __repr__(self):
        """String representation of current beta parameters."""
        return f"ThompsonSampling(successes={self.successes}, failures={self.failures})"

    def pull(self):
        """Choose an arm to pull using Thompson Sampling."""
        samples = np.random.beta(self.successes, self.failures)
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        """Update beta distribution parameters.

        Args:
          chosen_arm(int): Index of the selected arm.
          reward(float): Observed reward.

        Returns:

        """
        if reward > 0:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

    def experiment(self, num_trials=20000):
        """Run the Thompson Sampling experiment.

        Args:
          num_trials(int, optional): Number of trials to run. (Default value = 20000)

        Returns:

        """
        for _ in range(num_trials):
            chosen_arm = self.pull()
            reward = np.random.normal(self.p[chosen_arm], 1)
            self.update(chosen_arm, reward)
            self.rewards.append(reward)
            self.choices.append((chosen_arm, reward))

    def report(self):
        """Generate report: save rewards, print average reward and regret."""
        df = pd.DataFrame([(arm, r, 'ThompsonSampling') for arm, r in self.choices],
                          columns=['Bandit', 'Reward', 'Algorithm'])
        df.to_csv('data/rewards.csv', mode='a', index=False)
        avg_reward = np.mean(self.rewards)
        regret = np.sum(np.max(self.p) - np.array(self.rewards))
        logger.info(f"ThompsonSampling Avg Reward: {avg_reward:.4f}")
        logger.info(f"ThompsonSampling Cumulative Regret: {regret:.4f}")


def comparison():
    p = [1, 2]
    eg = EpsilonGreedy(p)
    ts = ThompsonSampling(p)

    eg.experiment()
    ts.experiment()

    eg.report()
    ts.report()

    vis = Visualization()
    vis.plot1(eg, ts)
    vis.plot2(eg, ts, p)
    vis.plot_distributions(p, [x[0] for x in eg.choices], [x[0] for x in ts.choices])


if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    logger.info("Starting bandit experiment comparison...")
    try:
        comparison()
        logger.success("Experiment completed and visualizations saved.")
    except Exception as e:
        logger.exception(f"Experiment failed due to: {e}")
