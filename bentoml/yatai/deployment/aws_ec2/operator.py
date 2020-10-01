"""
# make dockerfile of bentoservice
# push it to ecr
# make target group -> OPTIONAL
# make launch temaplate which does:
# pull image
# run it
# create security group which allows traffic on port 5000

# make autoscaling group for ec2 -> optional
# make template of all above and deploy it
"""
from yatai.deployment.operator import DeploymentOperatorBase
class AwsEc2DeploymentOperator(DeploymentOperatorBase):
    def add(self, deployment_pb):
        pass
    def update(self, deployment_pb):
        pass
    def delete(self, deployment_pb):
        pass
    def describe(self, deployment_pb):
        pass
