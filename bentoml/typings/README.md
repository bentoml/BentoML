Library stubs used by bentoml

Make sure to add an upstream remotes to bentoml main repository, find out more [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-for-a-fork)
If you want to contribute stubs that is not yet existed in our VCS, make sure to follow the below steps:

1. Generate stubs with `pyright`: `pyright --createstub <imports_library>`

2. Minimize stubs with `./scripts/tools/stubs_cleanup.sh`

3. Commit the typings repo, with `git add -f typings/<imports_library>`

4. Edit changes and get a diff files:
```bash
git diff HEAD upstream/main > <imports_library>.diff
```

5. Commit the diff files

Done.
