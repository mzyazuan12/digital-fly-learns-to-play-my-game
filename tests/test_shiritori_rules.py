from shiritori.dictionary import (
    build_dict,
    group_count,
    is_valid_word,
    playable_count,
    resolve_prefix_len,
)
from shiritori.prefix_blacklist import is_blocked_prefix
from shiritori.rules import desired_chain_len, turn_time_limit


TINY = build_dict(
    {
        "apple",
        "elephant",
        "tiger",
        "rabbit",
        "tree",
        "egg",
        "goat",
        "ant",
        "ear",
        "red",
        "dog",
        "grape",
    }
)


def test_word_membership():
    assert is_valid_word(TINY, "apple")
    assert not is_valid_word(TINY, "xyzzy")


def test_blocked_double_consonant_and_exact_list():
    assert is_blocked_prefix("mm")
    assert is_blocked_prefix("tion")
    assert not is_blocked_prefix("ti")
    assert not is_blocked_prefix("ofat")  # ofa is blocked; ofat is not


def test_resolve_prefix_shortens_when_needed():
    used = {"apple"}
    # "apple" ends in "e"; "elephant"/"egg"/"ear" remain.
    length = resolve_prefix_len(TINY, "apple", 4, used, min_groups=0, min_playable=1)
    assert 1 <= length <= 4
    assert "apple"[-length:]


def test_casual_group_count_positive_for_e():
    assert group_count(TINY, "e") >= 2
    assert playable_count(TINY, "e", {"egg"}, cap=10) >= 1


def test_chain_and_timer_match_live_game():
    assert desired_chain_len(0) == 1
    assert desired_chain_len(3) == 2
    assert desired_chain_len(11) == 4
    assert turn_time_limit(1) == 15
    assert turn_time_limit(70) == 5
