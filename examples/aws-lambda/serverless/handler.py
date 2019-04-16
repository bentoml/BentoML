import json
import os
import sys

try:
    import unzip_requirements
except ImportError:
    pass

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vendors'))

from vendors.TextClassificationService.v1_0_0 import TextClassificationService  # noqa: E402


bento_service = TextClassificationService()


def predict(event, context):
    input_data = json.loads(event['body'])

    result = bento_service.predict({'text': input_data.text})

    body = {
        "result": result[0][0]
    }
    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response
