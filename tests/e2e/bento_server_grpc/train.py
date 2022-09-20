if __name__ == "__main__":
    import python_model

    import bentoml

    bentoml.picklable_model.save_model(
        "py_model.case-1.grpc.e2e",
        python_model.PythonFunction(),
        signatures={
            "predict_file": {"batchable": True},
            "echo_json": {"batchable": True},
            "echo_object": {"batchable": False},
            "echo_ndarray": {"batchable": True},
            "double_ndarray": {"batchable": True},
            "multiply_float_ndarray": {"batchable": True},
            "double_dataframe_column": {"batchable": True},
        },
        external_modules=[python_model],
    )
