#!/usr/bin/env bash
set -Eeuo pipefail

# check to see if this file is being run or sourced from another script
_is_sourced() {
	# https://unix.stackexchange.com/a/215279
	[ "${#FUNCNAME[@]}" -ge 2 ] &&
		[ "${FUNCNAME[0]}" = '_is_sourced' ] &&
		[ "${FUNCNAME[1]}" = 'source' ]
}

_main() {
	# if first arg looks like a flag
	if [ -z "$@" ] || [ "${1:0:1}" = '-' ]; then
		if [[ -v YATAI_VERSION ]]; then
			if [ "$BENTOML_SERVER_COMPONENT" == "RUNNER" ]; then
				set -- python -m bentoml_cli.server.runner "$@" "$BENTO_PATH"
			else
				set -- python -m bentoml_cli.server.http_api_server "$@" "$BENTO_PATH"
			fi
		else
			set -- bentoml serve --production "$@" "$BENTO_PATH"
		fi
	fi

	# Overide the BENTOML_PORT if PORT env var is present. Used for Heroku **and Yatai**
	if [[ -v PORT ]]; then
		echo "\$PORT is set! Overiding \$BENTOML_PORT with \$PORT ($PORT)"
		export BENTOML_PORT=$PORT
	fi
	exec "$@"
}

if ! _is_sourced; then
	_main "$@"
fi
