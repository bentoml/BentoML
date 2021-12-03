### Instructions

pyright generates stubs from python imports. Thus there will be cases where library imports is different from the PyPI package:
  - Import namespace for `python_multipart` is `multipart`
  - Import namespace for `Pillow` is `PIL`
just to name a few...


To avoid this, when adding new library to stubs, make sure to add two entries into [imports.in](`./imports.in`) separated
 by a space:
```bash
<import_namespace> <pypi_package>
```

i.e: If you wants to add stubs for `localstack`, add the following line to [imports.in](`./imports.in`):
```bash
localstack localstack
```

Then generate given stubs by running [install.stubs.sh](`./install_stubs.sh`):
```bash
# Assumes current directory is "$GIT_ROOT"/scripts/stubs
./install_stubs.sh
```
