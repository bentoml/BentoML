from bentoml.handlers.utils import concat_list


def test_concat():
    lst = [
        [1],
        [1, 2],
        [],
        [1, 2, 3],
    ]
    datas, slices = concat_list(lst)

    for s, origin_data in zip(slices, lst):
        assert origin_data == datas[s]
