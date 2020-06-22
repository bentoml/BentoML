from bentoml.adapters.utils import concat_list


def test_concat():
    lst = [
        None,
        [],
        [1],
        [1, 2],
        [1, 2, 3],
    ]
    datas, slices = concat_list(lst)

    for s, origin_data in zip(slices, lst):
        if s is None:
            assert origin_data is None
        else:
            assert origin_data == datas[s]


def test_concat_with_flags():
    lst = [
        [1],
        None,
        1,
        None,
    ]
    flags = [
        True,
        True,
        False,
        False,
    ]

    datas, slices = concat_list(lst, flags)
    assert datas == [1, 1]

    for s, origin_data in zip(slices, lst):
        if s is None:
            assert origin_data is None
        else:
            assert origin_data == datas[s]


def test_concat_lists_with_flags():
    lst = [
        [[1], [2]],
        [],
        None,
        [1],
        "string",
        None,
    ]
    flags = [
        True,
        True,
        True,
        False,
        False,
        False,
    ]

    datas, slices = concat_list(lst, flags)
    assert datas == [[1], [2], [1], "string"]

    for s, origin_data in zip(slices, lst):
        if s is None:
            assert origin_data is None
        else:
            assert origin_data == datas[s]
