load("@bazel_skylib//rules:write_file.bzl", "write_file")

def run_shell(name, under_workspace = False, srcs = [], content = [], data = [], **kwargs):
    """
    Create a run_shell macro.
    We will create a shell wrapper, and then return a target
    that can be used to run the shell wrapper. The shell wrapper
    will run under $BUILD__DIRECTORY.
    Args:
        name: Name of the rule set.
        srcs: List of source files to be used by the rules.
        content: List of rules to be applied.
        data: List of data files to be used by the rules.
        under_workspace: Whether to run the shell wrapper under the BUILD_WORKSPACE_DIRECTORY. By default,
                         the shell wrapper will run under the BUILD_WORKING_DIRECTORY.
        **kwargs: Arbitrary keyword arguments.
    """
    file_name = "_{}_wrapper".format(name)
    shell_file = "{}.sh".format(file_name)
    if under_workspace:
        workspace = "$BUILD_WORKSPACE_DIRECTORY"
    else:
        workspace = "$BUILD_WORKING_DIRECTORY"

    write_file(
        name = file_name,
        out = shell_file,
        content = [
            "#!/usr/bin/env bash\n",
            "cd {}\n".format(workspace),
        ] + content,
    )
    native.sh_binary(
        name = name,
        srcs = [shell_file] + srcs,
        data = data,
        **kwargs
    )
