#!/usr/bin/env bash
# Runs on LocalStack boot (mounted into /etc/localstack/init/ready.d).
# Creates the buckets the pipeline expects so a fresh clone needs no manual setup.

set -euo pipefail

RAW_BUCKET="${S3_RAW_BUCKET:-quantfolio-raw}"
ARTIFACTS_BUCKET="${S3_ARTIFACTS_BUCKET:-quantfolio-artifacts}"

echo "[init-aws] creating buckets: ${RAW_BUCKET}, ${ARTIFACTS_BUCKET}"

awslocal s3api create-bucket --bucket "${RAW_BUCKET}" 2>/dev/null || true
awslocal s3api create-bucket --bucket "${ARTIFACTS_BUCKET}" 2>/dev/null || true

# Raw is the immutable record of what each source returned. Versioning means an
# accidental overwrite is recoverable rather than silently destructive.
awslocal s3api put-bucket-versioning \
  --bucket "${RAW_BUCKET}" \
  --versioning-configuration Status=Enabled

echo "[init-aws] buckets ready:"
awslocal s3 ls
