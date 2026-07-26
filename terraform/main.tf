/*
 * Infrastructure for QuantFolio, applied against LocalStack.
 *
 * These are not decorative configs — `terraform apply` genuinely creates the
 * buckets and you can list them afterwards with awslocal. Resources LocalStack
 * Community does not implement (ECR, Lambda, IAM enforcement) are kept in
 * ecs.tf behind a flag that defaults to off, so the default apply is honest:
 * everything it declares, it creates.
 *
 *   cd terraform
 *   terraform init
 *   terraform apply -auto-approve
 *   awslocal s3 ls
 */

terraform {
  required_version = ">= 1.5"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region     = var.aws_region
  access_key = "test"
  secret_key = "test"

  # LocalStack does not validate credentials or account IDs, and its S3
  # endpoint is path-style rather than virtual-hosted.
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true
  s3_use_path_style           = true

  endpoints {
    s3     = var.localstack_endpoint
    ecr    = var.localstack_endpoint
    lambda = var.localstack_endpoint
    iam    = var.localstack_endpoint
    logs   = var.localstack_endpoint
    sts    = var.localstack_endpoint
  }
}

# --------------------------------------------------------------------------- #
# Raw data: immutable, versioned, and never deleted by policy.
# --------------------------------------------------------------------------- #
resource "aws_s3_bucket" "raw" {
  bucket = var.raw_bucket

  tags = {
    Project     = "quantfolio"
    Purpose     = "immutable-raw-market-data"
    Environment = var.environment
  }
}

# Raw is the record of what each vendor actually returned. Versioning makes an
# accidental overwrite recoverable instead of silently destructive — the same
# property the ingestion code enforces in application logic.
resource "aws_s3_bucket_versioning" "raw" {
  bucket = aws_s3_bucket.raw.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_public_access_block" "raw" {
  bucket = aws_s3_bucket.raw.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Raw parquet compresses well but accumulates forever. Aging it into cheaper
# storage keeps the replay guarantee without paying hot-storage prices for
# 2019 data nobody reads.
resource "aws_s3_bucket_lifecycle_configuration" "raw" {
  bucket = aws_s3_bucket.raw.id

  rule {
    id     = "archive-old-raw"
    status = "Enabled"

    filter {
      prefix = "raw/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }

    transition {
      days          = 365
      storage_class = "GLACIER"
    }

    # Keep only recent overwrite history; the current version never expires.
    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }
}

# --------------------------------------------------------------------------- #
# Artifacts: MLflow models, staged parquet, portfolio weights.
# --------------------------------------------------------------------------- #
resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifacts_bucket

  tags = {
    Project     = "quantfolio"
    Purpose     = "mlflow-artifacts-and-staging"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Staging data is derived and rewritten every run, so it is disposable. Model
# artifacts under mlflow/ are not covered by this rule and persist.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "expire-staging"
    status = "Enabled"

    filter {
      prefix = "staging/"
    }

    expiration {
      days = 14
    }
  }
}
