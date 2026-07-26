/*
 * Compute definitions for the API service.
 *
 * Gated behind `enable_compute`, which defaults to false. LocalStack Community
 * does not implement ECR and Lambda reliably enough to claim these are really
 * created, and a config that "applies" without producing anything is worse than
 * one that says so.
 *
 * These are `terraform validate`-clean and are the real target shape for AWS:
 *
 *     terraform validate                          # always passes
 *     terraform apply -var enable_compute=true    # against real AWS
 */

# --------------------------------------------------------------------------- #
# Image registry
# --------------------------------------------------------------------------- #
resource "aws_ecr_repository" "api" {
  count = var.enable_compute ? 1 : 0

  name                 = "quantfolio-api"
  image_tag_mutability = "IMMUTABLE" # a deployed tag must never change meaning

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project     = "quantfolio"
    Environment = var.environment
  }
}

# Untagged layers accumulate on every rebuild and are pure cost.
resource "aws_ecr_lifecycle_policy" "api" {
  count = var.enable_compute ? 1 : 0

  repository = aws_ecr_repository.api[0].name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Keep the 10 most recent release images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = { type = "expire" }
      },
    ]
  })
}

# --------------------------------------------------------------------------- #
# Execution role
# --------------------------------------------------------------------------- #
resource "aws_iam_role" "api_execution" {
  count = var.enable_compute ? 1 : 0

  name = "quantfolio-api-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

# Read-only on artifacts, scoped to the two buckets. The API loads models; it
# has no reason to be able to write them, and no reason to see anything else.
resource "aws_iam_role_policy" "api_s3_read" {
  count = var.enable_compute ? 1 : 0

  name = "quantfolio-api-s3-read"
  role = aws_iam_role.api_execution[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*",
      ]
    }]
  })
}

# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #
resource "aws_lambda_function" "api" {
  count = var.enable_compute ? 1 : 0

  function_name = "quantfolio-api"
  role          = aws_iam_role.api_execution[0].arn
  package_type  = "Image"
  image_uri     = "${aws_ecr_repository.api[0].repository_url}:${var.api_image_tag}"

  # Generous because a cold start loads TensorFlow or PyTorch and pulls the
  # model bundle from S3; the steady-state request is far cheaper.
  timeout     = 60
  memory_size = 3008

  environment {
    variables = {
      MLFLOW_TRACKING_URI = "http://mlflow.internal:5000"
      S3_ARTIFACTS_BUCKET = aws_s3_bucket.artifacts.id
      S3_RAW_BUCKET       = aws_s3_bucket.raw.id
    }
  }

  tags = {
    Project     = "quantfolio"
    Environment = var.environment
  }
}

resource "aws_cloudwatch_log_group" "api" {
  count = var.enable_compute ? 1 : 0

  name              = "/aws/lambda/quantfolio-api"
  retention_in_days = 14
}
