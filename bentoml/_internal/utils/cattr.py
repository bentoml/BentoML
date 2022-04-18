from datetime import datetime

import cattr

bentoml_cattr = cattr.Converter()


def datetime_decoder(dt_like, _):
    if isinstance(dt_like, str):
        return datetime.fromisoformat(dt_like)
    elif isinstance(dt_like, datetime):
        return dt_like
    else:
        raise Exception(f"Unable to parse datetime from '{dt_like}'")


bentoml_cattr.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
bentoml_cattr.register_structure_hook(datetime, datetime_decoder)
