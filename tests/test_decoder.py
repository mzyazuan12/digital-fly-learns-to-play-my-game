import inspect

from flybrain.loader import synthetic_connectome
from flybrain.populations import ACTIONS, build_interface
from shiritori.action_decoder import KeyboardDecoder
from shiritori import action_decoder as decoder_mod


def test_decoder_has_no_word_list():
    source = inspect.getsource(decoder_mod)
    assert "by_prefix" not in source
    assert "words.txt" not in source
    assert "load_game_dict" not in source


def test_decoder_argmax_is_an_action_name():
    g = synthetic_connectome(n=96, extra_edges=20, seed=0)
    iface = build_interface(g, seed=0)
    dec = KeyboardDecoder(iface)
    counts = __import__("numpy").zeros(g.n, dtype=__import__("numpy").int32)
    # Force one action group to spike.
    counts[iface.action_groups[0]] = 10
    action = dec.decode(counts, 0.01)
    assert action in ACTIONS
