import os
import sys
import json
import shutil
import subprocess
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def check_aws_credentials_available():
    """
    Pre-boto3 checklist:
    1. Check for AWS environment variables (AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY)
    2. Check for ~/.aws/credentials or ~/.aws/config files
    3. Check for AWS container / ECS credentials in environment
    """
    # 1. Environment variables
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return True

    # 2. AWS Shared Credentials / Config files
    aws_creds_path = os.path.expanduser("~/.aws/credentials")
    aws_config_path = os.path.expanduser("~/.aws/config")
    if os.path.isfile(aws_creds_path) and os.path.getsize(aws_creds_path) > 0:
        return True
    if os.path.isfile(aws_config_path) and os.path.getsize(aws_config_path) > 0:
        return True

    # 3. Container / Web Identity / IAM Role variables
    if os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or \
       os.environ.get("AWS_CONTAINER_CREDENTIALS_FULL_URI") or \
       os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"):
        return True

    return False

def run_aws_login():
    """Trigger the environment's browser-based 'aws login' flow to (re)populate ~/.aws credentials."""
    if not shutil.which("aws"):
        print("Error: 'aws' CLI not found on PATH. Cannot run 'aws login'.")
        return False
    print("\n[AWS Pre-flight Checklist] Triggering 'aws login' to authenticate...")
    try:
        result = subprocess.run(["aws", "login"])
        return result.returncode == 0
    except (KeyboardInterrupt, subprocess.SubprocessError) as e:
        print(f"'aws login' failed: {e}")
        return False

def fetch_secret_value(secret_name, region):
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    if "SecretString" in response:
        return json.loads(response["SecretString"])
    return {}

def load_secrets(secret_name="LANGLAB_SECRETS", region_name=None):
    region = region_name or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-2"

    # Pre-boto3 checklist: verify credentials exist before attempting connection
    if not check_aws_credentials_available():
        if sys.stdin and sys.stdin.isatty():
            if not run_aws_login():
                return {}
        else:
            print(f"[AWS Pre-flight Checklist] No AWS credentials found for {secret_name}.")
            print("Tip: Mount ~/.aws, set AWS env vars, or run 'aws login' in a terminal.")
            return {}

    secrets = {}
    AUTH_ERROR_CODES = {
        "ExpiredToken", "ExpiredTokenException", "UnrecognizedClientException",
        "InvalidClientTokenId", "AccessDeniedException", "AuthFailure",
    }

    def is_auth_error(err):
        if isinstance(err, (NoCredentialsError, PartialCredentialsError)):
            return True
        if isinstance(err, ClientError):
            return err.response.get("Error", {}).get("Code") in AUTH_ERROR_CODES
        return "reauthenticate" in str(err).lower()

    try:
        secrets = fetch_secret_value(secret_name, region)
    except Exception as e:
        if is_auth_error(e) and sys.stdin and sys.stdin.isatty():
            print(f"AWS authentication issue: {e}")
            if run_aws_login():
                try:
                    secrets = fetch_secret_value(secret_name, region)
                except Exception as retry_err:
                    print(f"Error loading secrets after 'aws login': {retry_err}")
                    return {}
            else:
                return {}
        else:
            print(f"Error loading secrets from {secret_name}: {e}")
            return {}

    # Set environment variables in current process
    for key, value in secrets.items():
        os.environ[key] = str(value)

    # Write to .env file in workspace
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    with open(env_file, "w") as f:
        for key, value in secrets.items():
            f.write(f"{key}={value}\n")

    # Append export commands to ~/.bashrc for interactive shell sessions
    bashrc_file = os.path.expanduser("~/.bashrc")
    if os.path.exists(bashrc_file):
        with open(bashrc_file, "r") as f:
            content = f.read()
        marker = "# LANGLAB_SECRETS"
        if marker not in content:
            export_block = f"\n{marker}\n" + "\n".join([f'export {k}="{v}"' for k, v in secrets.items()]) + "\n"
            with open(bashrc_file, "a") as f:
                f.write(export_block)

    print(f"Successfully loaded {len(secrets)} secret(s) from {secret_name} into environment variables.")
    return secrets

if __name__ == "__main__":
    load_secrets()
