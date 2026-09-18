from shiritori.environment import ShiritoriEnv
from training.curriculum import tiny_words


def _env():
    return ShiritoriEnv.from_words(tiny_words(20), difficulty="casual", seed=0)


def test_reset_uses_dictionary_prefix():
    env = _env()
    obs = env.observe()
    assert obs["prefix"]
    assert env.last_word
    assert env.last_word in env.used


def test_valid_word_scores_and_bot_replies():
    env = _env()
    # Force a known board: last word apple, prefix e.
    env.used = {"apple"}
    env.last_word = "apple"
    env.prefix = "e"
    env.chain_len = 1
    env.buffer = ""
    for ch in "egg":
        env.step(ch)
    result = env.step("ENTER")
    assert result.info.get("event") == "valid_word"
    assert result.reward > 0
    assert env.score == 1
    # Bot should have played something unless the tiny set is exhausted.
    assert env.last_word != "egg" or result.info.get("bot_miss")


def test_invalid_word_costs_a_try():
    env = _env()
    env.prefix = "e"
    env.buffer = "eeeeeeee"
    tries = env.tries_left
    result = env.step("ENTER")
    assert result.info.get("event") == "fail"
    assert env.tries_left == tries - 1


def test_timeout_loses_a_life():
    env = _env()
    env.budget = 2
    env.step("NOOP")
    result = env.step("NOOP")
    # second step exceeds budget (turn_steps becomes 3? first increments to 1, second to 2)
    # With budget=2, turn_steps>2 triggers timeout. Need one more.
    if not result.info.get("reason") == "timeout":
        result = env.step("NOOP")
    assert result.info.get("reason") == "timeout" or result.info.get("life_lost")


def test_no_predetermined_word_required():
    env = _env()
    env.used = {"apple"}
    env.last_word = "apple"
    env.prefix = "e"
    # Both egg and elephant/ear/eagle are legal; the env must accept either.
    for word in ("egg", "ear"):
        env2 = _env()
        env2.used = {"apple"}
        env2.last_word = "apple"
        env2.prefix = "e"
        env2.chain_len = 1
        for ch in word:
            env2.step(ch)
        result = env2.step("ENTER")
        assert result.info.get("event") == "valid_word", word
