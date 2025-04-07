#!/bin/bash
set -e

# Configuration
ENVIRONMENT=${ENVIRONMENT:-dev}
AWS_REGION=${AWS_REGION:-us-west-2}
ECR_REPOSITORY=${ECR_REPOSITORY:-nces}
IMAGE_TAG=${IMAGE_TAG:-latest}
STACK_NAME="${ENVIRONMENT}-nces"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print usage
function usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -h, --help                 Show this help message"
  echo "  -e, --environment ENV      Set deployment environment (default: dev)"
  echo "  -r, --region REGION        Set AWS region (default: us-west-2)"
  echo "  -t, --tag TAG              Set image tag (default: latest)"
  echo "  --ecr-repo REPO            Set ECR repository name (default: nces)"
  echo "  --stack-name NAME          Set CloudFormation stack name (default: {env}-nces)"
  echo "  --create-ecr               Create ECR repository if it doesn't exist"
  echo "  --build-only               Only build and push the Docker image"
  echo "  --deploy-only              Only deploy the CloudFormation stack"
  echo "  --gpu                      Build and deploy with GPU support"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
      usage
      exit 0
      ;;
    -e|--environment)
      ENVIRONMENT="$2"
      STACK_NAME="${ENVIRONMENT}-nces"
      shift 2
      ;;
    -r|--region)
      AWS_REGION="$2"
      shift 2
      ;;
    -t|--tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --ecr-repo)
      ECR_REPOSITORY="$2"
      shift 2
      ;;
    --stack-name)
      STACK_NAME="$2"
      shift 2
      ;;
    --create-ecr)
      CREATE_ECR=true
      shift
      ;;
    --build-only)
      BUILD_ONLY=true
      shift
      ;;
    --deploy-only)
      DEPLOY_ONLY=true
      shift
      ;;
    --gpu)
      USE_GPU=true
      shift
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      usage
      exit 1
      ;;
  esac
done

# Set AWS region
echo -e "${YELLOW}Setting AWS region to ${AWS_REGION}${NC}"
aws configure set region ${AWS_REGION}

# Check if we should create ECR repository
if [ "$CREATE_ECR" = true ]; then
  echo -e "${YELLOW}Checking if ECR repository exists...${NC}"
  if ! aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} > /dev/null 2>&1; then
    echo -e "${YELLOW}Creating ECR repository ${ECR_REPOSITORY}...${NC}"
    aws ecr create-repository --repository-name ${ECR_REPOSITORY}
  else
    echo -e "${GREEN}ECR repository ${ECR_REPOSITORY} already exists.${NC}"
  fi
fi

# Build and push Docker image
if [ "$DEPLOY_ONLY" != true ]; then
  # Get ECR login
  echo -e "${YELLOW}Logging in to ECR...${NC}"
  aws ecr get-login-password | docker login --username AWS --password-stdin $(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com

  # Build Docker image
  echo -e "${YELLOW}Building Docker image...${NC}"
  if [ "$USE_GPU" = true ]; then
    echo -e "${YELLOW}Building with GPU support...${NC}"
    docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} -f Dockerfile.gpu .
  else
    docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .
  fi

  # Tag and push the image
  echo -e "${YELLOW}Tagging and pushing image to ECR...${NC}"
  docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} $(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
  docker push $(aws sts get-caller-identity --query Account --output text).dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
  
  echo -e "${GREEN}Successfully built and pushed image to ECR${NC}"
fi

# Deploy CloudFormation stack
if [ "$BUILD_ONLY" != true ]; then
  echo -e "${YELLOW}Deploying CloudFormation stack ${STACK_NAME}...${NC}"
  
  # Check if stack exists
  if aws cloudformation describe-stacks --stack-name ${STACK_NAME} > /dev/null 2>&1; then
    echo -e "${YELLOW}Updating existing stack...${NC}"
    aws cloudformation update-stack \
      --stack-name ${STACK_NAME} \
      --template-body file://cloudformation.yaml \
      --parameters \
        ParameterKey=EnvironmentName,ParameterValue=${ENVIRONMENT} \
        ParameterKey=ECRRepositoryName,ParameterValue=${ECR_REPOSITORY} \
        ParameterKey=ImageTag,ParameterValue=${IMAGE_TAG} \
      --capabilities CAPABILITY_IAM
    
    echo -e "${YELLOW}Waiting for stack update to complete...${NC}"
    aws cloudformation wait stack-update-complete --stack-name ${STACK_NAME}
  else
    echo -e "${YELLOW}Creating new stack...${NC}"
    aws cloudformation create-stack \
      --stack-name ${STACK_NAME} \
      --template-body file://cloudformation.yaml \
      --parameters \
        ParameterKey=EnvironmentName,ParameterValue=${ENVIRONMENT} \
        ParameterKey=ECRRepositoryName,ParameterValue=${ECR_REPOSITORY} \
        ParameterKey=ImageTag,ParameterValue=${IMAGE_TAG} \
      --capabilities CAPABILITY_IAM
    
    echo -e "${YELLOW}Waiting for stack creation to complete...${NC}"
    aws cloudformation wait stack-create-complete --stack-name ${STACK_NAME}
  fi
  
  # Get outputs
  echo -e "${GREEN}Deployment completed successfully!${NC}"
  echo -e "${YELLOW}Stack outputs:${NC}"
  aws cloudformation describe-stacks --stack-name ${STACK_NAME} --query "Stacks[0].Outputs" --output table
fi

echo -e "${GREEN}All done!${NC}"
