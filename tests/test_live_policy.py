from sim3d.live_policy import pick_live_word, remaining_words, should_give_up
from shiritori.dictionary import load_game_dict


def _dict():
    return load_game_dict(subset={"apple", "ear", "egg", "gate"}, letters_only=True)


def test_does_not_give_up_when_a_word_remains():
    d = _dict()
    used = {"apple"}
    assert not should_give_up(d, "e", used)
    assert pick_live_word(d, "e", used) in {"ear", "egg"}
    assert set(remaining_words(d, "e", used)) == {"ear", "egg"}


def test_gives_up_only_when_the_model_has_no_word():
    d = _dict()
    used = {"apple", "ear", "egg"}
    assert should_give_up(d, "e", used)
    assert pick_live_word(d, "e", used) is None


def test_empty_prefix_is_not_a_give_up():
    d = _dict()
    assert not should_give_up(d, "", set())
    assert not should_give_up(d, "   ", set())
