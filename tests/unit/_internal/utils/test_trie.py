from bentoml._internal.utils.trie import Trie


def test_trie():
    trie = Trie()

    assert not trie.contains("")
    assert not trie.contains("something")

    assert trie.insert("") == 0

    assert trie.contains("")
    assert not trie.contains("something")

    # if a trie contains an empty string the insert depth is still 1
    assert trie.insert("test") == 1
    assert trie.insert("test1") == 5
    assert trie.insert("a") == 1

    assert not trie.contains("t")
    assert not trie.contains("te")
    assert not trie.contains("tes")
    assert trie.contains("test")
    assert trie.contains("test1")
    assert trie.contains("a")
    assert not trie.contains("something else")

    trie = Trie("testword")

    assert trie.contains("testword")
    assert trie.insert("test") == 4
    assert trie.contains("test")
    assert trie.insert("some words") == 1
    assert trie.insert("some") == 4
