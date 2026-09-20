import gymnasium as gym

from agentforge.adapter.gymnasium import GymnasiumAdapter


class FakeEnvironment:
    def __init__(self):
        self.closed = False

    def reset(self, *, seed=None):
        return {"state": 0}, {"seed": seed}

    def step(self, action):
        return {"state": action}, 1.0, False, False, {"action": action}

    def close(self):
        self.closed = True


def test_is_gymnasium_env():
    adapter = GymnasiumAdapter(FakeEnvironment())
    assert isinstance(adapter, gym.Env)


def test_reset_contract():
    adapter = GymnasiumAdapter(FakeEnvironment())

    observation, info = adapter.reset(seed=7)

    assert observation == {"state": 0}
    assert info == {"seed": 7}


def test_step_contract():
    adapter = GymnasiumAdapter(FakeEnvironment())

    observation, reward, terminated, truncated, info = adapter.step(3)

    assert observation == {"state": 3}
    assert reward == 1.0
    assert terminated is False
    assert truncated is False
    assert info == {"action": 3}


def test_close_contract():
    environment = FakeEnvironment()
    adapter = GymnasiumAdapter(environment)

    adapter.close()

    assert environment.closed is True
