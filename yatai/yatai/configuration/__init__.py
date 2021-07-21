import os


def get_local_config_file():
    if "YATAI_CONFIG" in os.environ:
        # User local config file for customizing Yatai
        return expand_env_var(os.environ.get("YATAI_CONFIG"))
    return None


def inject_dependencies():
    """Inject dependencis and configuration for Yatai package"""

    from yatai.configuration.containers import YataiConfiguration, YataiContainer

    config_file = get_local_config_file()
    if config_file and config_file.endswith('.yml'):
        configuration = YataiConfiguration(override_config_file=config_file)
    else:
        configuration = YataiConfiguration()

    YataiContainer.config.set(configuration.as_dict())


def expand_env_var(env_var):
    """Expands potentially nested env var by repeatedly applying `expandvars` and
    `expanduser` until interpolation stops having any effect.
    """
    if not env_var:
        return env_var
    while True:
        interpolated = os.path.expanduser(os.path.expandvars(str(env_var)))
        if interpolated == env_var:
            return interpolated
        else:
            env_var = interpolated
