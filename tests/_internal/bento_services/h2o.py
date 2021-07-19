import typing as t

import h2o

if t.TYPE_CHECKING:
    import h2o.model
    import pandas as pd


def predict_dataframe(
    model: "h2o.model.model_base.ModelBase", df: "pd.DataFrame"
) -> t.Optional[str]:
    hf = h2o.H2OFrame(df)
    pred = model.predict(hf)
    return pred.as_data_frame().to_json(orient='records')
