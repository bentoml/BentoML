EC2_CLOUDFORMATION_TEMPLATE = """\
AWSTemplateFormatVersion: 2010-09-09
Transform: AWS::Serverless-2016-10-31
Description: BentoML load balanced template
Parameters:
    AmazonLinux2LatestAmiId:
        Type: AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>
        Default: {ami_id}
Resources:
    SecurityGroupResource:
        Type: AWS::EC2::SecurityGroup
        Properties:
            GroupDescription: "security group for bentoservice"
            SecurityGroupIngress:
                -
                    IpProtocol: tcp
                    CidrIp: 0.0.0.0/0
                    FromPort: 5000
                    ToPort: 5000
                -
                    IpProtocol: tcp
                    CidrIp: 0.0.0.0/0
                    FromPort: 22
                    ToPort: 22
            VpcId: !Ref Vpc1

    Ec2InstanceECRProfile:
        Type: AWS::IAM::InstanceProfile
        Properties:
            Path: /
            Roles: [!Ref EC2Role]

    EC2Role:
        Type: AWS::IAM::Role
        Properties:
            AssumeRolePolicyDocument:
                Statement:
                    -   Effect: Allow
                        Principal:
                            Service: [ec2.amazonaws.com]
                        Action: ['sts:AssumeRole']
            Path: /
            Policies:
                -   PolicyName: ecs-service
                    PolicyDocument:
                        Statement:
                            -   Effect: Allow
                                Action:
                                    -   'ecr:GetAuthorizationToken'
                                    -   'ecr:BatchGetImage'
                                    -   'ecr:GetDownloadUrlForLayer'
                                Resource: '*'

    LaunchTemplateResource:
        Type: AWS::EC2::LaunchTemplate
        Properties:
            LaunchTemplateName: {template_name}
            LaunchTemplateData:
                IamInstanceProfile:
                    Arn: !GetAtt Ec2InstanceECRProfile.Arn
                ImageId: !Ref AmazonLinux2LatestAmiId
                InstanceType: {instance_type}
                UserData: "{user_data}"
                SecurityGroupIds:
                - !GetAtt SecurityGroupResource.GroupId

    TargetGroup:
        Type: AWS::ElasticLoadBalancingV2::TargetGroup
        Properties:
            VpcId: !Ref Vpc1
            Protocol: HTTP
            Port: 5000
            TargetType: instance
            HealthCheckEnabled: true
            HealthCheckIntervalSeconds: {target_health_check_interval_seconds}
            HealthCheckPath: {target_health_check_path}
            HealthCheckPort: {target_health_check_port}
            HealthCheckProtocol: HTTP
            HealthCheckTimeoutSeconds: {target_health_check_timeout_seconds}
            HealthyThresholdCount: {target_health_check_threshold_count}

    LoadBalancerSecurityGroup:
        Type: AWS::EC2::SecurityGroup
        Properties:
            GroupDescription: "security group for loadbalancing"
            VpcId: !Ref Vpc1
            SecurityGroupIngress:
                -
                    IpProtocol: tcp
                    CidrIp: 0.0.0.0/0
                    FromPort: 80
                    ToPort: 80

    InternetGateway:
        Type: AWS::EC2::InternetGateway

    Gateway:
        Type: AWS::EC2::VPCGatewayAttachment
        Properties:
            InternetGatewayId: !Ref InternetGateway
            VpcId: !Ref Vpc1

    PublicRouteTable:
        Type: AWS::EC2::RouteTable
        Properties:
            VpcId: !Ref Vpc1

    PublicRoute:
        Type: AWS::EC2::Route
        DependsOn: Gateway
        Properties:
            DestinationCidrBlock: 0.0.0.0/0
            GatewayId: !Ref InternetGateway
            RouteTableId: !Ref PublicRouteTable

    RouteTableSubnetTwoAssociationOne:
        Type: AWS::EC2::SubnetRouteTableAssociation
        Properties:
          RouteTableId: !Ref PublicRouteTable
          SubnetId: !Ref Subnet1
    RouteTableSubnetTwoAssociationTwo:
        Type: AWS::EC2::SubnetRouteTableAssociation
        Properties:
          RouteTableId: !Ref PublicRouteTable
          SubnetId: !Ref Subnet2

    Vpc1:
        Type: AWS::EC2::VPC
        Properties:
            CidrBlock: 172.31.0.0/16
            EnableDnsHostnames: true
            EnableDnsSupport: true
            InstanceTenancy: default

    Subnet1:
        Type: AWS::EC2::Subnet
        Properties:
            VpcId: !Ref Vpc1
            AvailabilityZone:
                Fn::Select:
                    - 0
                    - Fn::GetAZs: ""
            CidrBlock: 172.31.16.0/20
            MapPublicIpOnLaunch: true

    Subnet2:
        Type: AWS::EC2::Subnet
        Properties:
            VpcId: !Ref Vpc1
            AvailabilityZone:
                Fn::Select:
                    - 1
                    - Fn::GetAZs: ""
            CidrBlock: 172.31.0.0/20
            MapPublicIpOnLaunch: true

    LoadBalancer:
        Type: AWS::ElasticLoadBalancingV2::LoadBalancer
        Properties:
            IpAddressType: ipv4
            Name: {elb_name}
            Scheme: internet-facing
            SecurityGroups:
                - !Ref LoadBalancerSecurityGroup
            Subnets:
                - !Ref Subnet1
                - !Ref Subnet2
            Type: application

    Listener:
        Type: AWS::ElasticLoadBalancingV2::Listener
        Properties:
            DefaultActions:
                -   Type: forward
                    TargetGroupArn: !Ref TargetGroup
            LoadBalancerArn: !Ref LoadBalancer
            Port: 80
            Protocol: HTTP

    AutoScalingGroup:
        Type: AWS::AutoScaling::AutoScalingGroup
        DependsOn: Gateway
        Properties:
            MinSize: {autoscaling_min_size}
            MaxSize: {autoscaling_max_size}
            DesiredCapacity: {autoscaling_desired_capacity}
            AvailabilityZones:
                - Fn::Select:
                    - 0
                    - Fn::GetAZs: ""
                - Fn::Select:
                    - 1
                    - Fn::GetAZs: ""
            LaunchTemplate:
                LaunchTemplateId: !Ref LaunchTemplateResource
                Version: !GetAtt LaunchTemplateResource.LatestVersionNumber
            TargetGroupARNs:
                - !Ref TargetGroup
            VPCZoneIdentifier:
            - !Ref Subnet1
            - !Ref Subnet2
        UpdatePolicy:
            AutoScalingReplacingUpdate:
                WillReplace: true

Outputs:
    S3Bucket:
        Value: {s3_bucket_name}
        Description: Bucket to store sam artifacts
    AutoScalingGroup:
        Value: !Ref AutoScalingGroup
        Description: Autoscaling group name
    TargetGroup:
        Value: !Ref TargetGroup
        Description: Target group for load balancer
    Url:
        Value: !Join ['', ['http://', !GetAtt [LoadBalancer, DNSName]]]
        Description: URL of the bento service

"""


EC2_USER_INIT_SCRIPT = """MIME-Version: 1.0
Content-Type: multipart/mixed; boundary=\"==MYBOUNDARY==\"

--==MYBOUNDARY==
Content-Type: text/cloud-config; charset=\"us-ascii\"

runcmd:

- sudo yum update -y
- sudo amazon-linux-extras install docker -y
- sudo service docker start
- sudo usermod -a -G docker ec2-user
- curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
- unzip awscliv2.zip
- sudo ./aws/install
- ln -s /usr/bin/aws aws
- aws ecr get-login-password --region {region}|docker login --username AWS --password-stdin {registry}
- docker pull {tag}
- docker run -p {bentoservice_port}:{bentoservice_port} {tag}

--==MYBOUNDARY==--
"""  # noqa: E501
