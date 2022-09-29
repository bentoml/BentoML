if __name__ == "__main__":
    import pickle_model

    import bentoml

    bentoml.picklable_model.save_model(
        "py_model.case-1.http.e2e",
        pickle_model.PickleModel(),
        signatures={
            "predict_file": {"batchable": True},
            "echo_json": {"batchable": True},
            "echo_obj": {"batchable": False},
            "echo_multi_ndarray": {"batchable": True},
            "predict_ndarray": {"batchable": True},
            "predict_multi_ndarray": {"batchable": True},
            "predict_dataframe": {"batchable": True},
        },
        external_modules=[pickle_model],
    )
