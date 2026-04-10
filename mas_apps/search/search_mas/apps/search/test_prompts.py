from search_mas.apps.search.prompts import SEARCH_PROMPT, VERIFIER_PROMPT


def test_verifier_prompt_does_not_render_step_text() -> None:
    prompt = VERIFIER_PROMPT.format(
        env_prompt="question",
        team_context='The output of "Verifier Agent": previous output',
        step=7,
    )

    assert "You are now at step 7." not in prompt


def test_search_prompt_does_not_render_step_text() -> None:
    prompt = SEARCH_PROMPT.format(
        env_prompt="question",
        team_context='The output of "Verifier Agent": previous output',
        step=7,
    )

    assert "# Your Teammates' Outputs at Step 7" not in prompt
    assert "# Your Teammates' Outputs\n" in prompt
