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
	# Overide the BENTOML_PORT if PORT env var is present. Used for Heroku
	if [[ -v PORT ]]; then
		echo "\\$PORT is set! Overiding \\$BENTOML_PORT with \\$PORT ($PORT)"
		export BENTOML_PORT=$PORT
	fi

	# if file /etc/arg_mamba_user exists, then we are using micromamba-docker
	if [[ -f /etc/arg_mamba_user ]]; then
		# get the user from the file
		mamba_user_file=$(cat /etc/arg_mamba_user)

		if [[ "${MAMBA_USER}" != "${mamba_user_file}" ]]; then
			echo "ERROR: This micromamba-docker image was built with" \
				"'ARG MAMBA_USER=${mamba_user_file}', but the corresponding" \
				"environment variable has been modified to 'MAMBA_USER=${MAMBA_USER}'." \
				"For instructions on how to properly change the username, please refer" \
				"to the documentation at <https://github.com/mamba-org/micromamba-docker>." >&2
			exit 1
		fi
		# if USER is not set and not root
		if [[ ! -v USER && $(id -u) -gt 0 ]]; then
			# should get here if 'docker run...' was passed -u with a numeric UID
			export USER="$MAMBA_USER"
			export HOME="/home/$USER"
		fi

		if [[ ! -f /usr/local/bin/_activate_current_env.sh ]]; then
			echo "ERROR: micromamba-docker removed or changed '_activate_current_env.sh' file." \
				"Please contact the BentoML team ASAP for further supports." >&2
			exit 1
		else
			source /usr/local/bin/_activate_current_env.sh
		fi
	fi

	exec "$@"
}

if ! _is_sourced; then
	_main "$@"
fi
