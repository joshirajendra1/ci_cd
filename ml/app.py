"""
Your module description
"""
import json

# import model and data module for ML

from model import PredictScore


def lambda_handler(event, context):
    """Sample pure Lambda function"""

    print(event)
    predicted_score = PredictScore(event)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {"message": "Your predicted output is " + str(predicted_score),}
        ),
    }
