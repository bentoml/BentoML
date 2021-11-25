from distutils.command.install_headers import install_headers as old_install_headers

class install_headers(old_install_headers):
    def run(self): ...
