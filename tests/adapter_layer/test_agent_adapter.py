from agentforge.adapter.agent import AgentAdapter


class FakeAgent:
    def act(self, observation):
        return {"action": observation}


def test_agent_adapter_forwards_act():
    adapter = AgentAdapter(FakeAgent())

    assert adapter.act({"state": 1}) == {
        "action": {"state": 1}
    }
