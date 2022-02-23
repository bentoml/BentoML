# Type stubs for BentoML

Workflow

- run `pyright --createstubs <library>`
- run `./scripts/tools/stubs_cleanup.sh`
- commit the stubs with --signoff with msg: chore(typings): add <library> stubs.
- after commit, `git format-patch -k --stdout HEAD~1 > typings/stubs-<library>.patch`
    - remove the stubs directory
- commit this changes in: message it: refactor(typings): add <library> patch.
- rebase with `git rebase -i -S --signoff HEAD~2`, squash into 1, only keep the
  second commit message
- `git push` to check in the patch file

when applying:
- `git apply --stat typings/stubs-<library>.patch`
- `git apply --check typings/stubs-<library>.patch`
- `git am --signoff < typings/stubs-<library>.patch`
