import os
import typing as t
import tempfile
import contextlib

import fasttext
from bentoml.fasttext import FastTextModel

test_json: t.Dict[str, str] = {"text": "foo"}


@contextlib.contextmanager
def _temp_filename_with_content(contents: t.Any) -> t.Generator[str, None, None]:
    temp_file = tempfile.NamedTemporaryFile(suffix=".txt", mode="w+")
    temp_file.write(contents)
    temp_file.seek(0)
    yield temp_file.name
    temp_file.close()


def test_fasttext_save_load(tmpdir):
    with _temp_filename_with_content("__label__bar foo") as inf:
        model = fasttext.train_supervised(input=inf)

    FastTextModel(model).save(tmpdir)
    assert os.path.isfile(os.path.join(tmpdir, "saved_model"))

    fasttext_loaded: "fasttext.FastText._FastText" = FastTextModel.load(tmpdir)
    assert fasttext_loaded.predict(test_json["text"])[0] == ("__label__bar",)
