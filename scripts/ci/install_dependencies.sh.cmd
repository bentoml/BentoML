:; if [ -z 0 ]; then
  goto :WINDOWS
fi

set -x
python -m pip install --upgrade pip
python -m pip install --upgrade --upgrade-strategy eager --editable ".[test]"
exit

:: cmd script
:WINDOWS
python -m pip install --upgrade pip
python -m pip install --upgrade --upgrade-strategy eager --editable ".[test]"
