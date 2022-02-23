# Type stubs for BentoML

This directory contains patch files for 3rd party stubs that is used by BentoML.
The reason for using `.patch` instead of checking-in the stubs itself is to
minimize the library size, as well as make it easier for new developers to
create and contains a consistent stubs for pyright.

# Instruction

To manually edit and fix a given library stubs, the workflows for creating a new
patch file are as follow:
- Run `pyright --createstubs <library>`.
- Run `./scripts/tools/clean_stubs.sh`.
- Commit the stubs with `--signoff` with msg: "chore(typings): add <library> stubs."
- After commit, `git format-patch -k --stdout HEAD~1 > typings/stubs-<library>.patch`
- Commit this changes with message: "refactor(typings): add <library> patch."
- Rebase with `git rebase -i -S --signoff HEAD~2`, squash into 1, only keep the second commit message.
- Add `typings/<library>` to `$GIT_ROOT/.gitignore`
- `git push` to check in the patch file.

To apply a patch file:
- `git apply --stat typings/stubs-<library>.patch`
- `git apply --check typings/stubs-<library>.patch`
- `git am --signoff < typings/stubs-<library>.patch`
