from agentforge.adapter.agentforge_bench import AgentForgeBenchAdapter


class FakeEnvironment:
    def __init__(self):
        self.closed = False

    def reset(self, *, seed=None):
        return {"seed": seed}

    def step(self, action):
        return {"action": action}

    def close(self):
        self.closed = True


def test_reset_forwards():
    adapter = AgentForgeBenchAdapter(FakeEnvironment())
    assert adapter.reset(seed=11) == {"seed": 11}


def test_step_forwards():
    adapter = AgentForgeBenchAdapter(FakeEnvironment())
    assert adapter.step("move") == {"action": "move"}


def test_close_forwards():
    environment = FakeEnvironment()
    adapter = AgentForgeBenchAdapter(environment)

    adapter.close()

    assert environment.closed is True
