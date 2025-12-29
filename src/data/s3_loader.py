import boto3
import pandas as pd
from io import StringIO


def load_titanic_from_s3(bucket_name: str, key: str) -> pd.DataFrame:
    """
    Load the Titanic dataset from S3 and return it as a pandas DataFrame.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket
    key : str
        Path to the CSV file inside the bucket

    Returns
    -------
    pd.DataFrame
        Titanic dataset
    """
    s3 = boto3.client("s3")

    response = s3.get_object(
        Bucket=bucket_name,
        Key=key
    )

    csv_bytes = response["Body"].read()
    csv_string = csv_bytes.decode("utf-8")

    df = pd.read_csv(StringIO(csv_string))
    return df
