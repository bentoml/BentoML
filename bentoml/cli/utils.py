import os
import shutil

from datetime import datetime


def save_deployment_archive(archive_path, current_saved_path, platform):
    saved_path = os.path.join(archive_path, '.bento-deployment', platform,
                              datetime.now().isoformat())
    shutil.copytree(current_saved_path, saved_path)
    shutil.rmtree(current_saved_path)
    return saved_path
