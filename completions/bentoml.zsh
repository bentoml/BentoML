#compdef bentoml

(( $+functions[__bentoml_override_options] )) ||
__bentoml_override_options() {
  # setup default opts
  autoload -U is-at-least

  unsetopt local_options menu_complete
  unsetopt local_options flowcontrol
  # Automatically list choices on ambiguous completion
  setopt local_options auto_list
  setopt local_options extended_glob
  setopt local_options globdots
  setopt local_options auto_menu
  setopt local_options complete_in_word
  setopt local_options always_to_end
}

## short option stacking can be enabled:
# zstyle ':completion:*:*:bentoml:*' option-stacking yes
# probably don't need to do this, but nice to have
#
# When developing, you can use the following to reload the completion after
# having modified it:
#
#    $ unfunction _bentoml && autoload -U _bentoml
# To display subcommand completion in groups, please add the following to your
# '.zshrc':
#
#    zstyle ":bentoml:*" use-groups true

##### Setup default options
(( $+functions[__bentoml_arguments] )) ||
__bentoml_arguments() {
  if zstyle -t ":completion:${curcontext}:" option-stacking; then
    print -- -s
  fi
}

(( $+functions[__bentoml_caching_policy] )) ||
__bentoml_caching_policy() {
  local -a oldp
  oldp=( "$1"(Nmh+7) )  # will be invalidated after 7h
  (( $#oldp ))
}

__bentoml_commands() {
  # https://stackoverflow.com/questions/70843900/zsh-completion-caching-policy-explained
  integer force_invalid=0

  local cache_policy
  local -a tmp models_group yatai_group bentoml_group

  zstyle -s ":completion:${curcontext}:*" cache-policy cache_policy
  if [[ -z "$cache_policy" ]]; then
    zstyle ":completion:${curcontext}:*" cache-policy __bentoml_caching_policy
  fi
  
  # handle cached completion for subcommands
  if ( [[ ${+_bentoml_subcommands} -eq 0 ]] || _cache_invalid bentoml_subcommands ) && ! _retrieve_cache bentoml_subcommands || [[ ${force_invalid} -eq 1 ]]; then
    local -a lines
    lines=(${(f)"$(_call_program commands bentoml 2>&1)"})
    # A: $((${lines[(i)*Commands:]} + 1)) -> get the index of the line of first commands. This line will be under Commands:
    # B: ${(M)${lines[A,-1]}:# *} -> get all of bentoml commands from command group, filter the group (https://zsh.sourceforge.io/Doc/Release/Expansion.html#Parameter-Expansion)
    # C: ${B## #} -> more filtering
    _bentoml_subcommands=(${${${(M)${lines[$((${lines[(i)*Commands:]} + 1)),-1]}:# *}## #}/ ##/:})
    # TODO: add ('help: Show help for a command') to _bentoml_subcommands for better CLI experience
    (( $#_bentoml_subcommands > 1 )) && _store_cache bentoml_subcommands _bentoml_subcommands
  fi

  # This takes care of alignment if the user wanted the subcommand completion
  # to be split into groups.
  models_group=(${(M)_bentoml_subcommands:#*models:Model*})
  yatai_group=(${(M)_bentoml_subcommands:#*yatai:Yatai*})
  bentoml_group=(${${_bentoml_subcommands:#*yatai:Yatai*}:#*models:Model*})

  zstyle -s ":bentoml:*" use-groups tmp
  if test -n "$tmp"; then
    _describe -t bentoml_group "BentoML commands" bentoml_group
    _describe -t bentoml_models_group "Model commands" models_group
    _describe -t bentoml_yatai_group "Yatai commands" yatai_group
  else
    _describe -t bentoml_parent_group "bentoml commands" bentoml_group -- models_group -- yatai_group
  fi
}

(( $+functions[__bentoml_completer_from_cmd] )) ||
__bentoml_completer_from_cmd() {
  [[ $PREFIX = -* ]] && return 1

  local cache_policy
  zstyle -s ":completion:${curcontext}:*" cache-policy cache_policy
  if [[ -z "$cache_policy" ]]; then
    zstyle ":completion:${curcontext}:*" cache-policy __bentoml_caching_policy
  fi

  integer ret=1
  declare -a items onlyitems lines
  local cmd type line m args

  cmd=$1; shift
  args=$1; shift
  type=$1; shift

  lines=(${${(f)${:-"$(_call_program commands $cmd $args)"$'\n'}}[2,-1]})

  case $cmd in
    docker)
      items=(${${${${${(k)lines}/ ##/:::}%% *}%:::<none>}#<none>})
      ;;
    bentoml)
      items=(${${${${(k)lines}/ ##/}/:/:::}%% *})
      ;;
  esac

  onlyitems=(${items%::*})

  for m in $onlyitems; do
    [[ ${PREFIX##${~~m}} != ${PREFIX} ]] && {
      items=(${${items/:::/:}/:/\\:})
      _describe -t $type-with-tags "$type with tags" items && ret=0
      return ret
    }
  done

  # only complete items
  onlyitems=(${${items%:::*}/:/\\:})
  _describe -t $type-only "$type" onlyitems -qS : && ret=0

  return ret
}

__bentoml_complete_model_tag() {
  [[ $PREFIX = -* ]] && return 1
  __bentoml_completer_from_cmd bentoml "models list" "models"
}

__bentoml_complete_bento_tag() {
  [[ $PREFIX = -* ]] && return 1
  __bentoml_completer_from_cmd bentoml "list" "bentos"
}

# modified from docker/docker-ce/components/cli/contrib/completion/zsh/_docker
__bentoml_complete_docker_repository() {
  [[ $PREFIX = -* ]] && return 1
  __bentoml_completer_from_cmd docker "images" "docker repositories"
}

__bentoml_export_paths() {
  [[ $PREFIX = -* ]] && return 1
  local -a tmp
  integer ret=1

  zstyle -s ":bentoml:*" use-groups tmp
  if test -n "$tmp"; then
    _values ':url paths:_urls'
    _values ':files:_path_files' && ret=0
  else
    _alternative ':url paths:_urls' ':files:_path_files' && ret=0
  fi
  return ret

}


__bentoml_store_abstraction_completer(){
  [[ $PREFIX = -* ]] && return 1

  integer ret=1
  local -a _command_args opts_help
  local help="-h --help"
  local complete_function

  complete_function=$1; shift

  opts_help=(
    "(: -)"{-h,--help}"[Show help and exit]"
    "($help --verbose --debug)"{--debug,--verbose}"[Generate debug information]"
    "($help -q --quiet)"{-q,--quiet}"[Suppress all warnings and info logs]"
    "($help)--do-not-track[Do not send usage info]"
    "($help)--config[BentoML configuration YAML file to apply]:config path:_files -g '*.yaml'"
  )

  case "$words[1]" in
    (delete)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help)"{-y,--yes,--assume-yes}"[Skip confirmation when deleting a specific bento bundle]" \
        "($help -):bento tag:$complete_function" && ret=0
      ;;
    (export)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -):bento tag:$complete_function" \
        "($help -)*:output path:__bentoml_export_paths" && ret=0
      ;;
    (get)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -o --output)"{-o,--output=}"[Specific output type for returned bento metadata]:type:(json yaml path)" \
        "($help -):bento tag:$complete_function" && ret=0
      ;;
    (ls|list)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -o --output)"{-o,--output=}"[Specific output type for listing bento]:type:(json yaml table)" \
        "($help)--no-trunc[Don't truncate outputs]:no truncate (text): " \
        "($help -):bento tag:$complete_function" && ret=0
      ;;
    (import)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -):input path:_files" && ret=0
      # TODO: support s3 auto complete here
      ;;
    (pull)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help)"{-f,--force}"[Force pull from yatai to local and overwrite even if the bento exists under local bento store]" \
        "($help -):bento tag:$complete_function" && ret=0
      ;;
    (push)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help)"{-f,--force}"[Force push from yatai to local and overwrite on yatai]" \
        "($help -):bento tag:$complete_function" && ret=0
      ;;
  esac

  return ret
}

__bentoml_models_commands() {
  # https://stackoverflow.com/questions/70843900/zsh-completion-caching-policy-explained
  local -a _bentoml_models_subcommands
  _bentoml_models_subcommands=(
    "delete:Delete Model in local model store"
    "export:Export a Model to an external archive file"
    "get:Print Model details by providing the model_tag"
    "import:Import a previously exported Model archive file"
    "list:List Models in local store"
    "pull:Pull Model from a Yatai server"
    "push:Push Model to a Yatai server"
  )

  _describe -t bentoml-models-commands "bentoml models commands" _bentoml_models_subcommands
}

__bentoml_yatai_commands() {
  local -a _bentoml_yatai_subcommands
  _bentoml_yatai_subcommands=("login:Login to Yatai server")

  _describe -t bentoml-yatai-commands "bentoml yatai commands" _bentoml_yatai_subcommands
}

###### Containerize autocompletion

__bentoml_complete_containerize_build_args() {
  [[ $PREFIX = -* ]] && return 1
  integer ret=1
  local -a buildkit_opts current_opts
  local current msg

  buildkit_opts=("BUILDKIT_CACHE_MOUNT_NS" "BUILDKIT_CONTEXT_KEEP_GIT_DIR" "BUILDKIT_INLINE_BUILDINFO_ATTRS" "BUILDKIT_INLINE_CACHE" "BUILDKIT_MULTI_PLATFORM" "BUILDKIT_SANDBOX_HOSTNAME" "BUILDKIT_SYNTAX")

  extracted=${${words[(r)-build-arg=*]}#*=}
  [[ "$extracted" != "" ]] && current_opts=("$extracted") || current_opts=(${(k)buildkit_opts})
  [[ "$extracted" != "" ]] && msg="<VARNAME>=<value>" || msg="BUILDKIT special build-arg"
  _describe -t local_with_opts_build_args $msg current_opts -qS = && ret=0

  return ret
}

__bentoml_complete_docker_buildx_builder() {
  [[ $PREFIX = -* ]] && return 1

  integer ret=1
  declare -a items lines
  local line

  lines=(${${(f)${:-"$(_call_program commands docker buildx ls)"$'\n'}}[2,-1]})
  items=(${${${(k)lines}:# *}/ * /})
  _describe -t bentoml_buildx_builder "buildx builder" items && ret=0
  return ret
}

__bentoml_completer_cache_type() {
  [[ $PREFIX = -* ]] && return 1
  integer ret=1
  local -a opts fullopts lines items
  local cache_type filter cmd execute

  cache_type=$1; shift
  [[ $cache_type = cache-to ]] && opts=("gha" "inline") || opts=("gha")
  fullopts=("${(k)opts}" "registry" "local")

  if compset -P "*="; then
    # TODO:
    # registry,ref=usr/repo:cache
    # local,src=/path/to/cache
    # NOTE: how should we handle this?
    filter=${${words[-1]#*=}#*=}
    if test -n "$filter"; then
      case $filter in
        (gha|inline)
          _message "No more arguments required" && ret=0
          ;;
        (registry)
          # lines=(${${(f)${:-"$(_call_program commands docker images)"$'\n'}}[2,-1]})
          # items=(${${${${${(k)lines}/ ##/:}%% *}%:<none>}#<none>})
          compadd -X "registry source:\n" -qS , && ret=0
          ;;
        (local)
          compadd -X "local source:\n" -qS , && ret=0
          ;;
      esac
    else
      _describe -t local_cache_type "$cache_type source" fullopts -q && ret=0
    fi
  fi
  return ret
}

__parsing_dockerfile_from(){return 0}
__parsing_dockerfile_copy(){return 0}

__bentoml_parsing_dockerfile() {
  [[ $PREFIX = -* ]] && return 1

  integer ret=1
  local -a stages froms copys files stages dockerfile_stages tmp
  local fname

  dockerfile_stages=()
  stages=("cached" "base" "base-arm64" "base-amd64" "base-386" "base-ppc64le" "base-s390x")

  # TODO: parsing build stages from Dockerfile
  files=${(f)$(find $PWD -iname '*[D|d]ockerfile*' -type f)}
  if test -n $files; then
    for f in $files; do
      fname=$(basename $f)
      copys=${(M)${(f)${:-"$(_call_program commands cat $f)"$'\n'}}:#COPY*}
      froms=${(M)${(f)${:-"$(_call_program commands cat $f)"$'\n'}}:#FROM*}
      dockerfile_stages+=(__parsing_dockerfile_from "$froms[@]")
      dockerfile_stages+=(__parsing_dockerfile_copy "$copys[@]")
      # _values -s , "$fname stages" "$dockerfile_stages[@]" && ret=0
      # _values -s , 'template bento stages' "$dockerfile_stages[@]" && ret=0
    done
  else
    _values -s , 'default bento stages' "$stages[@]" && ret=0
  fi

  return ret
}

__bentoml_commandgroup() {
  local -a _command_args opts_help
  local help="-h --help"
  integer ret=1

  opts_help=(
    "(: -)"{-h,--help}"[Show help and exit]"
    "($help --verbose --debug)"{--debug,--verbose}"[Generate debug information]"
    "($help -q --quiet)"{-q,--quiet}"[Suppress all warnings and info logs]"
    "($help)--do-not-track[Do not send usage info]"
    "($help)--config[BentoML configuration YAML file to apply]:config yaml:_files -g '*.yaml'"
  )

  case "$words[1]" in
    (build)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -f --bentofile)"{-f,--bentofile=}"[Path to bento build file, default to 'bentofile.yaml']:bentofile path:_files -g '*.yaml'" \
        "($help)--version=[Override default auto generated version string]:generated-bento-version: " \
        "($help -):build context (default to '.'):_directories" && ret=0
      ;;
    (containerize)
      # modified from docker/docker-ce/components/cli/contrib/completion/zsh/_docker
      local state

      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -t --docker-image-tag)*"{-t,--docker-image-tag=}"[Repository, name and optionally a tag (format: 'repo/name:tag'), defaults to bento tag.]: :__bentoml_complete_docker_repository" \
        "($help)*--add-host[Add a custom host-to-IP mapping (format: 'host:ip')]:host\:ip mapping: " \
        "($help)*--allow[Allow extra privileged entitlement (e.g., 'network.host', 'security.insecure')]:privileged entitlement:->entitlement" \
        "($help)*--build-arg=[Set build-time variables]: :->build_args" \
        "($help)*--build-context=[Additional build contexts (e.g, name=path)]:<varname>=<value>: " \
        "($help)*--cache-from=[External cache sources (e.g., 'user/app:cache', 'type=local,src=path/to/dir')]: :->cache_sources" \
        "($help)*--cache-to=[Exported cache destination (e.g., 'user/app:cache', 'type=local,src=path/to/dir')]: :->cache_sources" \
        "($help)*--label=[Set metadata for an image]:label=<value>: " \
        "($help)*--output=[Output destination (format: 'type=local,dest=path')]:type=local,dest=path; type=docker:(local tar oci docker image registry)" \
        "($help)*--secret=[Secret to expose to the build (format :'id=mysecret\[,src=/local/secret\]')]:[type=TYPE[,KEY=VALUE]: " \
        "($help)*--shm-size=[Size of '/dev/shm' (format is '<number><unit>')]:shm size: " \
        "($help)*--ssh=[SSH agent socket or keys to expose to the build (format: 'default\|<id>\[=<socket>|<key>\[,<key>\]\]')]:default|<id>\[=<socket>|<key>\[,<key>\]\]: " \
        "($help)*--ulimit=[ulimit options]:ulimit: " \
        "($help)--builder=[Override the configured builder instance]:builder instance:__bentoml_complete_docker_buildx_builder" \
        "($help)--platform=[Set target platform for build]:linux/amd64, linux/ppc64le: " \
        "($help)--cgroup-parent=[Optional parent cgroup for the container]:cgroup: " \
        "($help)--iidfile=[Write the image ID to the file]:image id file:_files -f" \
        "($help)--load[Shorthand for '--output=type=docker']" \
        "($help)--metadata-file=[Write build result metadata to a JSON file]:build metadata file:_files -f" \
        "($help)--network=[Set the networking mode for the 'RUN' instructions during build (default 'default')]:network: " \
        "($help)--no-cache[Do not use cache when building the image]" \
        "($help)--progress=[Set type of progress output ('auto', 'plain', 'tty'). Use plain to show container output (default to 'auto')]:progress:(auto plain tty)" \
        "($help)--pull[Always attempt to pull all referenced images]" \
        "($help)--push[Shorthand for '--output=type=registry']" \
        "($help)--no-cache-filter=[Do not cache specified stages]: :__bentoml_parsing_dockerfile" \
        "($help)--target=[Set the target build stage to build]: :__bentoml_parsing_dockerfile" \
        "($help -):bento tag:__bentoml_complete_bento_tag" && ret=0

      # TODO: add completion for each cache type
      case $state in
        (entitlement)
          [[ $PREFIX = -* ]] && return 1
          local -a opts
          opts=("network.host" "security.insecure")

          _describe -t docker_entitlement "Entitlement" opts && ret=0
          ;;
        (build_args)
          __bentoml_complete_containerize_build_args && ret=0
          ;;
        (cache_sources)
          [[ $PREFIX = -* ]] && return 1
          if [[ ${words[(r)--cache-from=type=*]} =~ --cache-from=type=* ]]; then
            __bentoml_completer_cache_type cache-from && ret=0
          elif [[ ${words[(r)--cache-to=type=*]} =~ --cache-to=type* ]]; then
            __bentoml_completer_cache_type cache-to && ret=0
          else
            __bentoml_complete_docker_repository && ret=0
          fi
          ;;
      esac
      ;;
    (delete|export|get|import|list|pull|push)
      __bentoml_store_abstraction_completer __bentoml_complete_bento_tag && ret=0
      ;;
    (serve)
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help)--production[Run BentoServer in production mode]" \
        "($help)--port[Port to listen on for REST API server, (default: 3000)]:port (int): " \
        "($help)--host[The host to bind for the REST api server (defaults: 127.0.0.1(dev), 0.0.0.0(production))]: host (0.0.0.0): " \
        "($help)--api-workers[Specify the number of API server workers to start. Default to number of available CPU cores in production mode]:num workers (int): " \
        "($help)--backlog[Number of pending connection, (default: 2048)]:backlog (int): " \
        "($help)--reload[Reload Service when code changes detected, this is only available in development mode]" \
        "($help)--reload-delay[Delay in seconds between each check if the Service needs to be reloaded  (default: 0.25)]: delay (float): " \
        "($help)--working-dir[When loading from source code, specify the directory to find the Service instance (default: .)]:working dir (str):_files -/" \
        "($help)"{--run-with-ngrok,--ngrok}"[Use ngrok to relay traffic on a public endpoint to the local BentoServer, only available in dev mode]" \
        "($help -):bento tag:{_alternative ':path:_files' __bentoml_complete_bento_tag}" && ret=0
      ;;
    (models)
      local curcontext="$curcontext" state
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -): :->command" \
        "($help -)*:: :->option-or-argument" && ret=0

      case $state in
        (command)
          __bentoml_models_commands && ret=0
          ;;
        (option-or-argument)
          curcontext=${curcontext%:*:*}:bentoml-$words[1]:
          case "$words[1]" in
            (delete|export|get|import|list|pull|push)
              __bentoml_store_abstraction_completer __bentoml_complete_model_tag && ret=0
              ;;
          esac
          ;;
      esac
      ;;
    (yatai)
      local curcontext="$curcontext" state
      _arguments $(__bentoml_arguments) -C \
        $opts_help \
        "($help -): :->command" \
        "($help -)*:: :->option-or-argument" && ret=0

      case $state in
        (command)
          __bentoml_yatai_commands && ret=0
          ;;
        (option-or-argument)
          curcontext=${curcontext%:*:*}:bentoml-$words[1]:
          case "$words[1]" in
            (login)
              _arguments $(__bentoml_arguments) -C \
                $opts_help \
                "($help)--endpoint[Yatai endpoint, i.e: https://yatai.com]:yatai endpoint: " \
                "($help)--api-token[Yatai user API token generated from dashboard]:api-token (hashed(str)): " && ret=0
              ;;
          esac
          ;;
      esac
      ;;
    (help)
      _arguments $(__bentoml_arguments) -C ":subcommand:__bentoml_commands" && ret=0
      ;;
  esac

  return ret
}

_bentoml() {
  local curcontext="$curcontext" state line help="-h --help"
  integer ret=1

  _arguments $(__bentoml_arguments) -C \
    "(: -)"{-h,--help}'[Show help and exit]' \
    "($help -v --version)"{-v,--version}"[Show BentoML version and exit]" \
    "($help -): :->commands" \
    "($help -)*:: :->option-or-argument" && ret=0

  case $state in
    (commands)
      __bentoml_commands && ret=0
      ;;
    (option-or-argument)
      # bentoml models li<tab> -> bentoml models list
      # curcontext have to shift from (:complete:bentoml:) -> (:complete:bentoml:*:*)
      curcontext=${curcontext%:*:*}:bentoml-$words[1]:
      __bentoml_commandgroup && ret=0
      ;;
  esac

  return ret
}

if [[ "$funcstack[1]" = "_bentoml" ]]; then
  # init override options
  __bentoml_override_options
  _bentoml "$@"
fi
