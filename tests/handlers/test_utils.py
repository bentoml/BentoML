from bentoml.adapters.utils import concat_list


def test_concat():
    lst = [
        [1],
        [1, 2],
        [],
        [1, 2, 3],
        None,
    ]
    datas, slices = concat_list(lst)

    for s, origin_data in zip(slices, lst):
        if s is None:
            assert origin_data is None
        else:
            assert origin_data == datas[s]

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

    for s, origin_data in zip(slices, lst):
        if s is None:
            assert origin_data is None
        else:
            assert origin_data == datas[s]
