from types import SimpleNamespace

import load_secrets
import pytest


def test_credentials_are_available_from_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-key")

    assert load_secrets.check_aws_credentials_available()


@pytest.mark.parametrize(
    "credential_variable",
    [
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
        "AWS_CONTAINER_CREDENTIALS_FULL_URI",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ],
)
def test_credentials_are_available_from_container_or_web_identity(monkeypatch, credential_variable):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setattr(load_secrets.os.path, "isfile", lambda _path: False)
    monkeypatch.setenv(credential_variable, "credential-location")

    assert load_secrets.check_aws_credentials_available()


def test_load_secrets_returns_empty_without_credentials_in_noninteractive_session(monkeypatch, capsys):
    monkeypatch.setattr(load_secrets, "check_aws_credentials_available", lambda: False)
    monkeypatch.setattr(load_secrets.sys, "stdin", SimpleNamespace(isatty=lambda: False))

    assert load_secrets.load_secrets("test-secret") == {}
    assert "No AWS credentials found for test-secret" in capsys.readouterr().out


def test_fetch_secret_value_reads_secret_string(monkeypatch):
    class FakeClient:
        def get_secret_value(self, SecretId):
            assert SecretId == "test-secret"
            return {"SecretString": '{"OPENAI_API_KEY": "test-key"}'}

    class FakeSession:
        def client(self, service_name, region_name):
            assert service_name == "secretsmanager"
            assert region_name == "us-east-1"
            return FakeClient()

    monkeypatch.setattr(load_secrets.boto3.session, "Session", FakeSession)

    assert load_secrets.fetch_secret_value("test-secret", "us-east-1") == {"OPENAI_API_KEY": "test-key"}


def test_load_secrets_writes_environment_file_without_touching_bashrc(monkeypatch, tmp_path):
    regions = []
    monkeypatch.setattr(load_secrets, "check_aws_credentials_available", lambda: True)
    monkeypatch.setattr(
        load_secrets,
        "fetch_secret_value",
        lambda _name, region: regions.append(region) or {"TEST_SETTING": "value"},
    )
    monkeypatch.setattr(load_secrets, "__file__", str(tmp_path / "load_secrets.py"))
    monkeypatch.setattr(load_secrets.os.path, "exists", lambda _path: False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)

    assert load_secrets.load_secrets("test-secret") == {"TEST_SETTING": "value"}
    assert regions == ["us-east-2"]
    assert load_secrets.os.environ["TEST_SETTING"] == "value"
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "TEST_SETTING=value\n"


def test_load_secrets_keeps_environment_values_when_env_file_write_fails(monkeypatch, capsys):
    monkeypatch.setattr(load_secrets, "check_aws_credentials_available", lambda: True)
    monkeypatch.setattr(load_secrets, "fetch_secret_value", lambda _name, _region: {"TEST_SETTING": "value"})
    monkeypatch.setattr(load_secrets.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(
        load_secrets,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only filesystem")),
        raising=False,
    )

    assert load_secrets.load_secrets("test-secret") == {"TEST_SETTING": "value"}
    assert load_secrets.os.environ["TEST_SETTING"] == "value"
    assert "could not be written" in capsys.readouterr().out


def test_run_aws_login_returns_false_when_cli_is_unavailable(monkeypatch):
    monkeypatch.setattr(load_secrets.shutil, "which", lambda _command: None)

    assert not load_secrets.run_aws_login()