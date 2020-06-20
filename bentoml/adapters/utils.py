def concat_list(lst, batch_flags=None):
    """
    >>> lst = [
        [1],
        [1, 2],
        [1, 2, 3],
        None,
        ]
    >>> concat_list(lst)
    [1, 1, 2, 1, 2, 3], [slice(0, 1), slice(1, 3), slice(3, 6), None]
    """
    slices = [slice(0)] * len(lst)
    datas = []
    row_flag = 0
    for i, r in enumerate(lst):
        if r is None:
            slices[i] = None
            continue
        j = -1
        if batch_flags is None or batch_flags[i]:
            for j, d in enumerate(r):
                datas.append(d)
            slices[i] = slice(row_flag, row_flag + j + 1)
        else:
            datas.append(r)
            slices[i] = row_flag
        row_flag += j + 1
    return datas, slices
