#!/bin/sh

echo "Setting the .env vars"

# APPEND VARIABLES TO '.env' FILE

# Prevents writing in the last non-empty line
echo "" >> .env

echo "AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID" >> .env
echo "AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY" >> .env
echo "AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION" >> .env

