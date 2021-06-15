import pathlib
import sys
import time

import bentoml
from bentoml.adapters import JsonInput


@bentoml.env(infer_pip_packages=True)
class ExampleService(bentoml.BentoService):
    """
    Example BentoService class made for testing purpose
    """

    @bentoml.api(input=JsonInput(), mb_max_latency=3 * 1000, batch=True)
    def echo_with_delay_max3(self, input_datas):
        data = input_datas[0]
        time.sleep(float(data))
        return input_datas


if __name__ == "__main__":
    artifacts_path = sys.argv[1]
    bento_dist_path = sys.argv[2]
    service = ExampleService()

    pathlib.Path(bento_dist_path).mkdir(parents=True, exist_ok=True)
    service.save_to_dir(bento_dist_path)
