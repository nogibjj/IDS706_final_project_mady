import subprocess


def run_command(command):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,  # Ensures strings instead of bytes
            encoding="utf-8",  # Specifies UTF-8 encoding
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        raise


def push_docker_image_to_ecr(image_name, tag, repository_name, aws_account_id, region):
    """Push a Docker image to AWS ECR."""
    ecr_uri = "381492212823.dkr.ecr.us-west-2.amazonaws.com/mady_ids706_final_proj"
    dockerfile_path = "../Dockerfile"  # Adjust this path
    build_context = ".."  # Adjust this path

    # Authenticate Docker with ECR
    print("Authenticating with ECR...")
    run_command(
        f"aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 381492212823.dkr.ecr.us-west-2.amazonaws.com"
    )

    # Build Docker image
    print(f"Building Docker image {image_name}:{tag}...")
    run_command(
        f"docker build -t {image_name}:{tag} -f {dockerfile_path} {build_context}"
    )

    # Tag Docker image
    print(f"Tagging Docker image as {ecr_uri}:{tag}...")
    run_command(f"docker tag {image_name}:{tag} {ecr_uri}:{tag}")

    # Push Docker image to ECR
    print(f"Pushing Docker image to ECR repository {repository_name}...")
    run_command(f"docker push {ecr_uri}:{tag}")

    print("Docker image pushed successfully!")


# Replace these with your details
IMAGE_NAME = "mady_final_project"
TAG = "latest"
REPOSITORY_NAME = "mady_ids706_final_proj"
AWS_ACCOUNT_ID = "381492212823"
REGION = "us-west-2"

# Push the Docker image
push_docker_image_to_ecr(IMAGE_NAME, TAG, REPOSITORY_NAME, AWS_ACCOUNT_ID, REGION)
