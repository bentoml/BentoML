#!/bin/bash
# The user for BentoML ec2 deployment will require the following IAM permissions to function properly:
#   AmazonEC2FullAccess
#   AmazonEC2ContainerRegistryFullAccess 
#   AmazonS3FullAccess
#   IAMFullAccess
#   AmazonVPCFullAccess
#   AWSCloudFormationFullAccess 
#   CloudWatchFullAccess
#   ElasticLoadBalancingFullAccess 
#   AutoScalingFullAccess

if [ "$#" -eq 2 ]; then
  GROUP_NAME=$1
  USER_NAME=$2
else
  echo "No GROUP_NAME or USER_NAME supplied"
  exit 1
fi

DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

aws iam create-group --group-name $GROUP_NAME

POLICY_ARN=$(aws iam create-policy --policy-name ssm-policy-9 --policy-document file://"$DIR"/ssm_policy.json | jq '.Policy.Arn' | sed -e 's/^"//' -e 's/"$//')
aws iam attach-group-policy --policy-arn $POLICY_ARN --group-name $GROUP_NAME

aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/IAMFullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AmazonVPCFullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AWSCloudFormationFullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/ElasticLoadBalancingFullAccess --group-name $GROUP_NAME
aws iam attach-group-policy --policy-arn arn:aws:iam::aws:policy/AutoScalingFullAccess --group-name $GROUP_NAME

aws iam create-user --user-name $USER_NAME
aws iam add-user-to-group --user-name $USER_NAME --group-name $GROUP_NAME
aws iam create-access-key --user-name $USER_NAME