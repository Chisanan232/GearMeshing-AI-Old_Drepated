from gearmeshing_ai.mcp_client.policy import Policy


def test_policy_default_allow() -> None:
    p: Policy = Policy()
    assert p.can_call("dev", "anything").allowed is True


def test_policy_deny_missing_tool_for_role() -> None:
    p: Policy = Policy({"prod": {"safe_tool"}}, default_allow=False)
    assert p.can_call("prod", "unsafe_tool").allowed is False
