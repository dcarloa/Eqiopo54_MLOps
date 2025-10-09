#!/usr/bin/env bash
# Configure AWS CLI from .env file safely

set -e  # Exit immediately if any command fails
set -o pipefail  # Catch errors in pipelines

# Load .env file
if [ ! -f .env ]; then
  echo "❌ .env file not found. Please create one with AWS credentials."
  exit 1
fi

# Export environment variables, ignoring comments and empty lines
set -a
source .env
set +a

# Validate required variables
: "${AWS_ACCESS_KEY_ID:?❌ Missing AWS_ACCESS_KEY_ID in .env}"
: "${AWS_SECRET_ACCESS_KEY:?❌ Missing AWS_SECRET_ACCESS_KEY in .env}"

# Optional defaults
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
AWS_OUTPUT="${AWS_OUTPUT:-json}"

# Configure AWS CLI
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
aws configure set region "$AWS_DEFAULT_REGION"
aws configure set output "$AWS_OUTPUT"

# Verify credentials and show identity
echo "🔍 Verifying AWS credentials..."
if identity_json=$(aws sts get-caller-identity 2>/dev/null); then
  echo "✅ AWS CLI configured successfully."
  echo "🧾 AWS Caller Identity:"
  echo "$identity_json" | jq
else
  echo "❌ AWS CLI configuration failed. Please check your credentials."
  exit 1
fi