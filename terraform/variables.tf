variable "aws_region" {
  description = "AWS region. LocalStack ignores it, but the provider requires one."
  type        = string
  default     = "us-east-1"
}

variable "localstack_endpoint" {
  description = "LocalStack edge endpoint. Point at real AWS by clearing the endpoints block."
  type        = string
  default     = "http://localhost:4566"
}

variable "environment" {
  description = "Environment tag applied to every resource."
  type        = string
  default     = "local"
}

variable "raw_bucket" {
  description = "Bucket for immutable raw vendor responses."
  type        = string
  default     = "quantfolio-raw"
}

variable "artifacts_bucket" {
  description = "Bucket for MLflow artifacts and staged parquet."
  type        = string
  default     = "quantfolio-artifacts"
}

variable "enable_compute" {
  description = <<-EOT
    Create the ECR repository and Lambda service definition.

    Defaults to false because LocalStack Community does not implement these
    reliably. Turning it on against real AWS creates them for real; leaving it
    off keeps `terraform apply` honest about what it actually built.
  EOT
  type        = bool
  default     = false
}

variable "api_image_tag" {
  description = "Container image tag for the API service."
  type        = string
  default     = "latest"
}
