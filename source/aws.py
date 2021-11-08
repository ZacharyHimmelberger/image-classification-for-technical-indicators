""" 
This script allows the user to connect to AWS, create an Amazon s3 bucket, and send files to the bucket.

There is also a function to remove a file locally.
"""

import pathlib as Path

import boto3
from botocore.exceptions import ClientError

from source import plots


def get_s3_resource(region='us-east-2'):
    """Returns a boto3 s3 resource with a specific region.

    Args:
        region (str, optional): Region for s3 resource. Defaults to 'us-east-2'.

    Returns:
        An s3 resource object.
    """
    s3_resource = boto3.resource(service_name='s3', region_name=region)   

    return s3_resource


def create_bucket(bucket_name, region='us-east-2'):
    """Create an S3 bucket with the specified name and in the specified region.

    Args:
        bucket_name (str): Name of bucket being created.
        region (str, optional): Region for s3 resource. Defaults to 'us-east-2'.

    Returns:
        An s3 resource object.
    """
    s3_resource = get_s3_resource(region)

    try:
        s3_resource.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={'LocationConstraint': region})

        return s3_resource
    
    except ClientError as e:
        print(f'Unexpected error: {e}')
        print('Bucket not created.')


def to_bucket(bucket_name, file, s3_resource=None, region='us-east-2'):
    """Uploads a specified file to the specified bucket in a specified region.

    Args:
        bucket_name (str): Name of bucket to send file.
        file (str): Relative filepath of file to be sent.
        s3_resource (str, optional): An s3 resource object. Defaults to None.
        region (str, optional): Region for s3 resource. Defaults to 'us-east-2'.
    """
    if s3_resource is None:
        s3_resource = get_s3_resource(region)

    s3_resource.Bucket(bucket_name).upload_file(Filename=file, Key=file)


def del_file(file):
    """Removeds a file from the directory.

    Args:
        file (str): Relative filepath of file to be removed.
    """
    if Path(file).is_file():
        Path.unlink(file)


def send_dir(bucket_name, dir_name, name='df.gzip'):
    """Uploads all files in a specified directory to a specified AWS bucket.

    Args:
        bucket_name (str): Name of AWS bucket to send.
        dir_name (str): Name of directory.
        name (str, optional): Name of h2o file. Defaults to 'df.gzip'.
    """
    p = Path(dir_name)
    list_of_files = [f.name for f in p.iterdir() if (f.is_file()) & ~(f.name.startswith('.'))]
    h2o_df = plots.build_h2o_dataset(list_of_files, [i for i in range(len(list_of_files))])

    for filename in list_of_files:
        to_bucket(bucket_name, filename)
        del_file(filename)
    
    h2o_df.to_parquet(f'{dir_name}{name}')
    to_bucket(bucket_name, f'{dir_name}{name}')
    del_file(f'{dir_name}{name}')
