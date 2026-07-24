import pytest

from bentoml._internal.utils.filesystem import resolve_user_filepath


def test_resolve_relative_to_ctx_outside_cwd(tmp_path, monkeypatch):
    """A file that is legitimately relative to ctx must resolve even when the
    current working directory is somewhere else.

    Regression test for the ``bentoml containerize`` failure where a
    ``dockerfile_template`` relative to the Bento build context was rejected
    with "Accessing file outside of current working directory is not allowed"
    simply because the process cwd differed from the build context.
    """
    build_ctx = tmp_path / "myproject"
    build_ctx.mkdir()
    template = build_ctx / "Dockerfile.template"
    template.write_text("FROM debian\n")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = resolve_user_filepath("Dockerfile.template", ctx=str(build_ctx))
    assert resolved == str(template.resolve())


def test_resolve_escaping_ctx_is_blocked(tmp_path, monkeypatch):
    """Paths that escape the base directory are still rejected under secure."""
    build_ctx = tmp_path / "myproject"
    build_ctx.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("secret\n")

    monkeypatch.chdir(build_ctx)

    with pytest.raises(ValueError, match="outside of the base directory"):
        resolve_user_filepath("../secret.txt", ctx=str(build_ctx))


def test_resolve_hidden_file_is_blocked(tmp_path, monkeypatch):
    """Hidden segments relative to the base directory are still rejected."""
    build_ctx = tmp_path / "myproject"
    build_ctx.mkdir()
    hidden = build_ctx / ".env"
    hidden.write_text("SECRET=1\n")

    monkeypatch.chdir(build_ctx)

    with pytest.raises(ValueError, match="hidden files"):
        resolve_user_filepath(".env", ctx=str(build_ctx))


def test_resolve_dotted_base_dir_not_false_positive(tmp_path, monkeypatch):
    """A base directory that itself lives under a dotted segment (e.g. a temp
    dir or ~/.cache) must not trigger the hidden-file check for files that are
    not themselves hidden relative to that base."""
    build_ctx = tmp_path / ".cache" / "myproject"
    build_ctx.mkdir(parents=True)
    template = build_ctx / "Dockerfile.template"
    template.write_text("FROM debian\n")

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = resolve_user_filepath("Dockerfile.template", ctx=str(build_ctx))
    assert resolved == str(template.resolve())


def test_resolve_absolute_path_blocked_under_secure(tmp_path, monkeypatch):
    """Absolute paths are still rejected under secure mode."""
    build_ctx = tmp_path / "myproject"
    build_ctx.mkdir()
    template = build_ctx / "Dockerfile.template"
    template.write_text("FROM debian\n")

    monkeypatch.chdir(build_ctx)

    with pytest.raises(ValueError, match="is not allowed"):
        resolve_user_filepath(str(template.resolve()), ctx=str(build_ctx))


def test_resolve_insecure_allows_outside(tmp_path, monkeypatch):
    """With secure=False, paths outside the base directory are permitted."""
    build_ctx = tmp_path / "myproject"
    build_ctx.mkdir()
    other = tmp_path / "other.txt"
    other.write_text("data\n")

    monkeypatch.chdir(build_ctx)

    resolved = resolve_user_filepath(
        str(other.resolve()), ctx=str(build_ctx), secure=False
    )
    assert resolved == str(other.resolve())
