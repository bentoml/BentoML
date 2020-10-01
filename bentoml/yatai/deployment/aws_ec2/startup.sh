MIME-Version: 1.0
Content-Type: multipart/mixed; boundary="==MYBOUNDARY=="

--==MYBOUNDARY==
Content-Type: text/cloud-config; charset="us-ascii"

runcmd:

- sudo yum update -y
- sudo amazon-linux-extras install docker -y
- sudo service docker start
- sudo usermod -a -G docker ec2-user
- curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
- unzip awscliv2.zip
- sudo ./aws/install
- ln -s /usr/bin/aws aws

- export AWS_ACCESS_KEY_ID=
- export AWS_SECRET_ACCESS_KEY=

- aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 752014255238.dkr.ecr.ap-south-1.amazonaws.com
- sudo docker pull 752014255238.dkr.ecr.ap-south-1.amazonaws.com/bento-iris:latest
- sudo docker run -p 5000:5000 752014255238.dkr.ecr.ap-south-1.amazonaws.com/bento-iris:latest

--==MYBOUNDARY==--