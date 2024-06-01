import click
from typer.testing import CliRunner
import pytest
import os
from pathlib import Path
from ..main import install
from pytest_httpx import HTTPXMock

runner = CliRunner()


def get_test_resource(name: str) -> Path:
    return Path(os.path.join(os.path.dirname(__file__), "testresources", name))


def test_install_invalid_archive(tmp_path):
    data = b"data"
    file_path = tmp_path / "test.tar"
    with open(file_path, "wb") as f:
        f.write(data)
    with pytest.raises(click.exceptions.Exit):
        install(
            file_path,
            ["https://example.com"],
            cache=False,
            force=False,
            start_on_boot=False,
        )
    assert os.listdir(tmp_path) == ["test.tar"]


def test_install(tmp_path, httpx_mock: HTTPXMock):
    httpx_mock.add_response(
        method="POST", json={"state": "success", "detail": "installed"}
    )
    time_skill = get_test_resource("time_example")
    try:
        install(
            time_skill.as_posix(),
            ["https://example.com"],
            cache=False,
            force=False,
            start_on_boot=False,
        )
    except click.exceptions.Exit as e:
        assert e.exit_code == 0
