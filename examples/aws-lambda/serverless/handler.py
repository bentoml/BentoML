import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'packages'))

from TextClassificationService.v1_0_0 import TextClassificationService  # noqa: E402


bento_service = TextClassificationService()


def predict(event, context):
    print(event, context)
    body = {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "input": event
    }

    result = bento_service.predict({'text': event.text})

    response = {
        "statusCode": 200,
        "body": json.dumps(body)
    }

    return response

    # Use this code if you don't use the http event with the LAMBDA-PROXY
    # integration
    """
    return {
        "message": "Go Serverless v1.0! Your function executed successfully!",
        "event": event
    }
    """
