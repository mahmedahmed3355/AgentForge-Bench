from agentforge.agents.rl import Policy, RLAgent


class DummyPolicy(Policy):
    def act(self, observation):
        return observation


class DummyAgent(RLAgent):
    def __init__(self):
        self.reset_count = 0
        self.last_observation = None

    def reset(self, *, seed=None, options=None):
        self.reset_count += 1

    def act(self, observation):
        return observation

    def observe(
        self,
        observation,
        *,
        reward=None,
        terminated=False,
        truncated=False,
        info=None,
    ):
        self.last_observation = observation


def test_rl_agent_contract():
    agent = DummyAgent()

    agent.reset(seed=7)
    action = agent.act({"state": 1})
    agent.observe(
        {"state": 2},
        reward=1.0,
        terminated=False,
        truncated=False,
        info={},
    )
    agent.finish(terminated=True, info={})

    assert agent.reset_count == 1
    assert action == {"state": 1}
    assert agent.last_observation == {"state": 2}
    assert agent.metadata() == {}


def test_policy_contract():
    policy = DummyPolicy()

    policy.reset(seed=7)
    assert policy.act({"state": 3}) == {"state": 3}
    assert policy.metadata() == {}


def test_rl_agent_and_policy_are_distinct_contracts():
    assert RLAgent is not Policy
