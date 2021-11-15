:; if [ -z 0 ]; then
  goto :WINDOWS
fi

set -x
python -m pip install --upgrade pip
python -m pip install --upgrade --editable ".[test]"
exit

:: cmd script
:WINDOWS
python -m pip install --upgrade pip
python -m pip install --upgrade --editable ".[test]"
