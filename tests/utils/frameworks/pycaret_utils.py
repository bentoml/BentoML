from pycaret.datasets import get_data


def get_pycaret_data():
    dataset = get_data("credit")
    data = dataset.sample(frac=0.95, random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(inplace=True, drop=True)
    data_unseen.reset_index(inplace=True, drop=True)

    return data, data_unseen[:5]
