import os
import shutil
import sys
import zipfile


pkgdir = '/tmp/bento'

sys.path.append(pkgdir)

if not os.path.exists(pkgdir):
    tempdir = '/tmp/_bento'
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

    default_lambda_task_root = os.environ.get('LAMBDA_TASK_ROOT', os.getcwd())
    lambda_task_root = os.getcwd() if os.environ.get('IS_LOCAL') == 'true' else default_lambda_task_root
    zip_requirements = os.path.join(lambda_task_root, '.requirements.zip')

    zipfile.ZipFile(zip_requirements, 'r').extractall(tempdir)
    os.rename(tempdir, pkgdir)
