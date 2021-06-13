#!/bin/sh

# the purpose of this scripts:
# - setup minikube, kubectl, helm if doesn't exists through binary
# - Run through most of configurations setup for k8s cluster.
#
# NOTES:
# - This scripts should run as-is, meaning this script will run within BentoML documents scope.
# - Example BentoService: aarnphm/bentoml-sentiment-analysis:latest
# - Assume that users before running this script to setup VirtualBox
# - This only works on POSIX system as we will install binary for most of dependency (skip if users already have dependencies)

# Contains functions from https://github.com/client9/shlib
set -e

LOG_LEVEL=2

GIT_ROOT=$(git rev-parse --show-toplevel)
K8S_CONFIG="${GIT_ROOT}/docs/source/guides/configs"
BIN_DIR=/usr/local/bin

TMP_DIR=$(mktemp -d)

trap 'rm -rf ${TMP_DIR}' EXIT

usage () {
  this="$1"
  cat <<EOF
${this}: Setup BentoService on K8s with Prometheus-Grafana stack

Usage: ${this} [-d] [-h]
  -d enables debug logging. (use this if you want to see the setup in action.)
  -h show this help message
EOF
	exit 2
}

main() {
  parse_args "$@"

  _NAMESPACE=bentoml
  if ! is_command virtualbox; then
    log_err "In order to run the script, you need to install virtualbox. Exitting."
    return 1
  fi
  log_info "While this scripts help ease the some of the pain for the demo, it is recommend for users to go through every step provided by the guide."

  log_info "Setting up binary..."
  install_binary
  cd "${K8S_CONFIG}"

  log_info "Setting up minikube..."
  if [ "$(minikube config get driver)" != "virtualbox" ]; then
      minikube config set driver virtualbox
  fi
  minikube delete && minikube start \
      --kubernetes-version=v1.20.0 \
      --memory=6g --bootstrapper=kubeadm \
      --extra-config=kubelet.authentication-token-webhook=true \
      --extra-config=kubelet.authorization-mode=Webhook \
      --extra-config=scheduler.address=0.0.0.0 \
      --extra-config=controller-manager.address=0.0.0.0

  log_info "setting up helm chart"
  helm repo list | grep prometheus >/dev/null || helm repo add prometheus-community https://prometheus-community.github.io/helm-charts

  # We will apply patches for both prometheus and grafana services.
  helm install prometheus-community/kube-prometheus-stack --create-namespace \
  --namespace "${_NAMESPACE}" --generate-name \
  --set prometheus.service.type=NodePort \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false

  log_info "patching our grafana services..."
  _GRAFANA_SVC=$(kubectl get svc -n "${_NAMESPACE}" | grep grafana | cut -d " " -f1)
  kubectl patch svc "${_GRAFANA_SVC}" -n "${_NAMESPACE}" --patch "$(cat deployment/grafana-patch.yaml)"

  log_info "setting up our BentoService..."
  kubectl apply -f deployment/bentoml-deployment.yml --namespace=bentoml

  log_debug "TODO: Added PersistentVolume for both Grafana (https://github.com/grafana/helm-charts/tree/main/charts/grafana#configuration) \
   and Prometheus Operator (https://github.com/prometheus-operator/prometheus-operator)."

  log_info "Done."
  return
}

install_binary() {

  _OS=$(get_os)
  _ARCH=$(get_arch)

  if [ "${_OS}" = "windows" ]; then
    log_crit "windows is not supported while running this script."
    return 1
  fi
  log_info "This scripts will ask for your sudo password for installing binary."

  ## minikube
  if ! is_command minikube; then
    _MINIKUBE_SCRIPTS="${TMP_DIR}/minikube"
    _MINIKUBE_BIN="${BIN_DIR}/minikube"
    log_info "Installing minikube to ${_MINIKUBE_BIN}"
    http_download "${_MINIKUBE_SCRIPTS}" "https://storage.googleapis.com/minikube/releases/latest/minikube-${_OS}-${_ARCH}" || exit 1
    sudo install "${_MINIKUBE_SCRIPTS}" "${_MINIKUBE_BIN}"
  fi

  ## kubectl
  if ! is_command kubectl; then
    _KUBECTL_SCRIPTS="${TMP_DIR}/kubectl"
    _KUBECTL_BIN="${BIN_DIR}/kubectl"
    _CHECKSUM="${TMP_DIR}/kubectl.sha256"
    log_info "Installing kubectl to ${_KUBECTL_BIN}"
    http_download "${_KUBECTL_SCRIPTS}" "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/${_OS}/${_ARCH}/kubectl" || exit 1

    # validate checksum
    log_info "Validating kubectl checksum..."
    http_download "${_CHECKSUM}" "https://dl.k8s.io/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/${_OS}/${_ARCH}/kubectl.sha256" || exit 1
    hash_sha256_verify "${_KUBECTL_SCRIPTS}" "${_CHECKSUM}"

    # install as root
    if [ "${_OS}" = "darwin" ]; then
      chmod +x "${_KUBECTL_SCRIPTS}"
      sudo mv "${_KUBECTL_SCRIPTS}" "${_KUBECTL_BIN}"
      sudo chown root: "${_KUBECTL_BIN}"
    else
      sudo install -o root -g root -m 0755 "${_KUBECTL_SCRIPTS}" "${_KUBECTL_BIN}"
    fi
  fi

  ## Helm
  if ! is_command helm; then
    log_info "Installing helm..."
    _HELM_SCRIPTS="${TMP_DIR}/get_helm.sh"
    http_download "${_HELM_SCRIPTS}" https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 || exit 1
    # shellcheck disable=SC1090
    chmod 700 "${_HELM_SCRIPTS}" && . "${_HELM_SCRIPTS}"
  fi

  return
}


#####################
# Helpers functions #
#####################

parse_args() {
  while getopts ":dh?t:" arg; do
    case "${arg}" in
    d)
      set -x
      LOG_LEVEL=3
      ;;
    h | \?)
      usage "$0"
      ;;
    *)
      return 1
      ;;
    esac
  done
}

get_os() {
	os=$(uname -s | tr '[:upper:]' '[:lower:]')
	case "${os}" in
	cygwin_nt*) os="windows" ;;
	mingw*) os="windows" ;;
	msys_nt*) os="windows" ;;
	*) os="${os}" ;;
	esac
	echo "${os}"
}

get_arch() {
	arch=$(uname -m)
	case "${arch}" in
	386) arch="i386" ;;
	aarch64) arch="arm64" ;;
	armv*) arch="arm" ;;
	i386) arch="i386" ;;
	i686) arch="i386" ;;
	x86) arch="i386" ;;
	x86_64) arch="amd64" ;;
	*) arch="${arch}" ;;
	esac
	echo "${arch}"
}

http_download_curl() {
	local_file=$1
	source_url=$2
	header=$3
	if [ -z "${header}" ]; then
		code=$(curl -w '%{http_code}' -sL -o "${local_file}" "${source_url}")
	else
		code=$(curl -w '%{http_code}' -sL -H "${header}" -o "${local_file}" "${source_url}")
	fi
	if [ "${code}" != "200" ]; then
		log_debug "http_download_curl received HTTP status ${code}"
		return 1
	fi
	return 0
}

http_download_wget() {
	local_file=$1
	source_url=$2
	header=$3
	if [ -z "${header}" ]; then
		wget -q -O "${local_file}" "${source_url}" || return 1
	else
		wget -q --header "${header}" -O "${local_file}" "${source_url}" || return 1
	fi
}

http_download() {
	log_debug "http_download $2"
	if is_command curl; then
		http_download_curl "$@" || return 1
		return
	elif is_command wget; then
		http_download_wget "$@" || return 1
		return
	fi
	log_crit "http_download unable to find wget or curl"
	return 1
}

hash_sha256() {
	target=$1
	if is_command sha256sum; then
		hash=$(sha256sum "${target}") || return 1
		echo "${hash}" | cut -d ' ' -f 1
	elif is_command shasum; then
		hash=$(shasum -a 256 "${target}" 2>/dev/null) || return 1
		echo "${hash}" | cut -d ' ' -f 1
	elif is_command sha256; then
		hash=$(sha256 -q "${target}" 2>/dev/null) || return 1
		echo "${hash}" | cut -d ' ' -f 1
	elif is_command openssl; then
		hash=$(openssl dgst -sha256 "${target}") || return 1
		echo "${hash}" | cut -d ' ' -f a
	else
		log_crit "hash_sha256 unable to find command to compute SHA256 hash"
		return 1
	fi
}

hash_sha256_verify() {
	target=$1
	checksums=$2
	basename=${target##*/}

	want=$(grep "${basename}" "${checksums}" 2>/dev/null | tr '\t' ' ' | cut -d ' ' -f 1)
	if [ -z "${want}" ]; then
		log_err "hash_sha256_verify unable to find checksum for ${target} in ${checksums}"
		return 1
	fi

	got=$(hash_sha256 "${target}")
	if [ "${want}" != "${got}" ]; then
		log_err "hash_sha256_verify checksum for ${target} did not verify ${want} vs ${got}"
		return 1
	fi
}

is_command() {
	command -v "$1" >/dev/null
}

log_debug() {
	[ 3 -le "${LOG_LEVEL}" ] || return 0
	echo DEBUG "$@" 1>&2
}

log_info() {
	[ 2 -le "${LOG_LEVEL}" ] || return 0
	echo INFO "$@" 1>&2
}

log_err() {
	[ 1 -le "${LOG_LEVEL}" ] || return 0
	echo ERROR "$@" 1>&2
}

log_crit() {
	[ 0 -le "${LOG_LEVEL}" ] || return 0
	echo CRITICAL "$@" 1>&2
}

main "$@"