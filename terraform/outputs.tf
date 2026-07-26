output "raw_bucket" {
  description = "Immutable raw market data bucket."
  value       = aws_s3_bucket.raw.id
}

output "artifacts_bucket" {
  description = "MLflow artifacts and staging bucket."
  value       = aws_s3_bucket.artifacts.id
}

output "raw_versioning_status" {
  description = "Proof that the replay guarantee is enforced at the bucket level."
  value       = aws_s3_bucket_versioning.raw.versioning_configuration[0].status
}

output "verify_command" {
  description = "Run this after apply to see the buckets Terraform created."
  value       = "awslocal s3 ls"
}

output "compute_enabled" {
  description = "Whether the ECR/Lambda definitions were applied."
  value       = var.enable_compute
}
