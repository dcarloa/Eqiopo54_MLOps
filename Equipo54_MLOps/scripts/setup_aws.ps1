# configure_aws.ps1
# Load environment variables and configure AWS CLI

$envFile = ".env"
if (-Not (Test-Path $envFile)) {
    Write-Host ".env file not found. Please create one with AWS credentials."
    exit 1
}

# Read .env file and set environment variables
Get-Content $envFile | ForEach-Object {
    if ($_ -match '^\s*#') { return } # skip comments
    if ($_ -match '^\s*$') { return } # skip blank lines
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
        $key = $parts[0].Trim()
        $value = $parts[1].Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value)
    }
}

# Configure AWS CLI
aws configure set aws_access_key_id $env:AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $env:AWS_SECRET_ACCESS_KEY

$region = if ($env:AWS_DEFAULT_REGION) { $env:AWS_DEFAULT_REGION } else { "us-east-1" }
aws configure set region $region
aws configure set output json

# Verify caller identity
Write-Host "Checking AWS identity..."
aws sts get-caller-identity

Write-Host "AWS CLI configured successfully."
