# Personalize Workshop

## Setting up Personalize from the console:

### Create a bucket in S3 to host the user-item and user csv files.
### Run code in push_to_s3.ipynb notebook and input your own bucket url where required.

#### Add the following bucket policy to the bucket you created for this workshop (go to bucket Permissions and paste the following json file).
{
    "Version": "2012-10-17",
    "Id": "PersonalizeS3BucketAccessPolicy",
    "Statement": [
        {
            "Sid": "PersonalizeS3BucketAccessPolicy",
            "Effect": "Allow",
            "Principal": {
                "Service": "personalize.amazonaws.com"
            },
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::pedro-workshop-hello",
                "arn:aws:s3:::pedro-workshop-hello/*"
            ]
        }
    ]
}

### Open personalize on the console.
### Create new Dataset group
### Create Dataset with existing schema
### Create Dataset import job
### Create new IAM role:
#### Attach IAMFullAccess, AmazonS3FullAccess and AmazonPersonalizeFullAccess policies 
#### Add the following inline policy to the IAM role:
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iam:AttachRolePolicy",
                "iam:CreateRole",
                "iam:CreateInstanceProfile",
                "iam:UpdateAssumeRolePolicy",
                "iam:AddRoleToInstanceProfile",
                "iam:PassRole",
                "iam:PutRolePolicy"
            ],
            "Resource": "*"
        }
    ]
}

#### Add the following trust policy to the IAM role:
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "personalize.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}

### When dataset import job is active > Create new solution

### When Solution is created > Launch Campaign
