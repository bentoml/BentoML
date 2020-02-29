## BentoML Cloud Deployment End-to-end Tests

### Prerequisite:
* Docker
* BentoML
* AWS CLI
* AWS Sam CLI
* AWS Credentials must be configured

### How to run test:
1. pip install -r requirements.txt
2. pip install -e . # install BentoML in dev mode
3. pytest e2e_tests
