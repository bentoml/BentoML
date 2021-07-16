import bentoml
from tests.bento_services.easyocr_service import EasyOCRService
from bentoml.yatai.client import YataiClient

import imageio
import easyocr

TEST_RESULT = ['西', '愚园路', '东', '315', '309', 'W', 'Yuyuan Rd。', 'E']
IMAGE_PATH = "./tests/integration/chinese.jpg"


def test_easyocr_artifact_packs():
    svc = EasyOCRService()

    lang_list = ['ch_sim', 'en']
    recog_network = "zh_sim_g2"

    model = easyocr.Reader(
        lang_list=lang_list,
        gpu=False,
        download_enabled=True,
        recog_network=recog_network,
    )
    svc.pack('chinese_small', model, lang_list=lang_list, recog_network=recog_network)

    assert [x[1] for x in model.readtext(IMAGE_PATH)] == (
        TEST_RESULT
    ), 'Run inference before saving the artifact'

    saved_path = svc.save()
    loaded_svc = bentoml.load(saved_path)

    assert loaded_svc.predict(imageio.imread(IMAGE_PATH))['text'] == (
        TEST_RESULT
    ), 'Run inference after saving the artifact'

    # clean up saved bundle
    yc = YataiClient()
    yc.repository.delete(f'{svc.name}:{svc.version}')
