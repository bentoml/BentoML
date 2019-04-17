import subprocess


def generate_base_serverless_files(output_path, platform, name):
    subprocess.call(
        ['serverless', 'create', '--template', platform, '--path', output_path, '--name', name])
    return


def add_model_service_archive(archive_path):
    return


def generate_handler_py(output_path):
    return
