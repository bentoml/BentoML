from typing import Any
from yatai.yatai.cli.bento_management import add_bento_management_sub_commands as add_bento_management_sub_commands
from yatai.yatai.cli.label import add_label_sub_commands as add_label_sub_commands
from yatai.yatai.cli.server import add_yatai_service_sub_commands as add_yatai_service_sub_commands

logger: Any

def create_yatai_cli_group(): ...
def create_yatai_cli(): ...
