"""Tests for daemon_client module."""

import os
import subprocess
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

from hindsight_embed import daemon_client
from hindsight_embed.daemon_embed_manager import DaemonEmbedManager


@pytest.fixture
def config():
    """Default config for tests."""
    return {
        "llm_api_key": "test-key",
        "llm_provider": "openai",
        "llm_model": "gpt-4o-mini",
        "bank_id": "test-bank",
    }

@pytest.fixture
def mock_cli_binary(tmp_path):
    """Create a mock CLI binary."""
    cli_path = tmp_path / "hindsight"
    cli_path.write_text("#!/bin/bash\nexit 0")
    cli_path.chmod(0o755)
    return cli_path

class TestRunCli:
    """Tests for run_cli function."""

    def test_run_cli_with_external_api_url(self, config, mock_cli_binary, monkeypatch):
        """Test that external HINDSIGHT_EMBED_API_URL skips daemon startup."""
        # Set up environment with external API URL
        external_api_url = "http://external-api:8000"
        monkeypatch.setenv("HINDSIGHT_EMBED_API_URL", external_api_url)

        # Mock functions
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=True)
        mock_subprocess_run = Mock(return_value=Mock(returncode=0))

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
            patch("subprocess.run", mock_subprocess_run),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon was NOT started (since external API URL is set)
            assert mock_ensure_daemon.call_count == 0

            # Verify CLI was called
            assert mock_subprocess_run.call_count == 1
            call_args = mock_subprocess_run.call_args

            # Verify environment contains the external API URL
            assert call_args.kwargs["env"]["HINDSIGHT_API_URL"] == external_api_url

            # Verify exit code
            assert exit_code == 0

    def test_run_cli_without_external_api_url(self, config, mock_cli_binary, monkeypatch):
        """Test that without external API URL, daemon is started."""
        # Ensure HINDSIGHT_EMBED_API_URL is not set
        monkeypatch.delenv("HINDSIGHT_EMBED_API_URL", raising=False)

        # Mock functions
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=True)
        mock_subprocess_run = Mock(return_value=Mock(returncode=0))

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
            patch("subprocess.run", mock_subprocess_run),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon WAS started (since no external API URL)
            assert mock_ensure_daemon.call_count == 1
            assert mock_ensure_daemon.call_args[0][0] == config

            # Verify CLI was called
            assert mock_subprocess_run.call_count == 1
            call_args = mock_subprocess_run.call_args

            # Verify environment contains the local daemon URL
            assert call_args.kwargs["env"]["HINDSIGHT_API_URL"] == daemon_client.get_daemon_url()

            # Verify exit code
            assert exit_code == 0

    def test_run_cli_daemon_startup_failure(self, config, mock_cli_binary, monkeypatch):
        """Test that daemon startup failure is handled properly."""
        # Ensure HINDSIGHT_EMBED_API_URL is not set
        monkeypatch.delenv("HINDSIGHT_EMBED_API_URL", raising=False)

        # Mock functions - daemon startup fails
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=False)  # Daemon fails to start

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon startup was attempted
            assert mock_ensure_daemon.call_count == 1

            # Verify exit code indicates failure
            assert exit_code == 1

    def test_run_cli_without_cli_binary(self, config, monkeypatch):
        """Test that missing CLI binary is handled properly."""
        # Ensure HINDSIGHT_EMBED_API_URL is not set
        monkeypatch.delenv("HINDSIGHT_EMBED_API_URL", raising=False)

        # Mock functions - CLI not installed
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=None)  # CLI not found

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify exit code indicates failure
            assert exit_code == 1

    def test_run_cli_with_api_token(self, config, mock_cli_binary, monkeypatch):
        """Test that HINDSIGHT_EMBED_API_TOKEN is passed through to the CLI."""
        # Set up environment with external API URL and token
        external_api_url = "http://external-api:8000"
        api_token = "test-bearer-token-12345"
        monkeypatch.setenv("HINDSIGHT_EMBED_API_URL", external_api_url)
        monkeypatch.setenv("HINDSIGHT_EMBED_API_TOKEN", api_token)

        # Mock functions
        mock_ensure_cli = Mock(return_value=True)
        mock_find_cli = Mock(return_value=mock_cli_binary)
        mock_ensure_daemon = Mock(return_value=True)
        mock_subprocess_run = Mock(return_value=Mock(returncode=0))

        with (
            patch.object(daemon_client, "ensure_cli_installed", mock_ensure_cli),
            patch.object(daemon_client, "find_cli_binary", mock_find_cli),
            patch.object(daemon_client, "ensure_daemon_running", mock_ensure_daemon),
            patch("subprocess.run", mock_subprocess_run),
        ):
            # Run CLI
            exit_code = daemon_client.run_cli(["memory", "recall", "test", "query"], config)

            # Verify daemon was NOT started (since external API URL is set)
            assert mock_ensure_daemon.call_count == 0

            # Verify CLI was called
            assert mock_subprocess_run.call_count == 1
            call_args = mock_subprocess_run.call_args

            # Verify environment contains both the API URL and the API key
            assert call_args.kwargs["env"]["HINDSIGHT_API_URL"] == external_api_url
            assert call_args.kwargs["env"]["HINDSIGHT_API_KEY"] == api_token

            # Verify exit code
            assert exit_code == 0


class TestClearPort:
    """Tests for DaemonEmbedManager._clear_port."""

    def test_port_free(self):
        """Port not in use — returns True immediately."""
        manager = DaemonEmbedManager()
        with patch.object(DaemonEmbedManager, "_is_port_in_use", return_value=False):
            assert manager._clear_port(9555) is True

    def test_port_occupied_by_hindsight_reuses_it_by_default(self):
        """Port occupied by a healthy hindsight daemon is reused by default."""
        manager = DaemonEmbedManager()
        with (
            patch.object(DaemonEmbedManager, "_is_port_in_use", return_value=True),
            patch("httpx.Client") as mock_httpx_cls,
            patch.object(DaemonEmbedManager, "_find_pid_on_port") as mock_find_pid,
            patch.object(DaemonEmbedManager, "_kill_process") as mock_kill,
        ):
            mock_client = MagicMock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = Mock(status_code=200)
            mock_httpx_cls.return_value = mock_client

            assert manager._clear_port(9555) is True
            mock_find_pid.assert_not_called()
            mock_kill.assert_not_called()

    def test_port_occupied_by_hindsight_replace_existing_stops_it(self):
        """Explicit replace kills the healthy daemon already serving the port."""
        manager = DaemonEmbedManager()
        with (
            patch.object(DaemonEmbedManager, "_is_port_in_use", return_value=True),
            patch("httpx.Client") as mock_httpx_cls,
            patch.object(DaemonEmbedManager, "_find_pid_on_port", return_value=12345),
            patch.object(DaemonEmbedManager, "_kill_process", return_value=True),
        ):
            mock_client = MagicMock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = Mock(status_code=200)
            mock_httpx_cls.return_value = mock_client

            assert manager._clear_port(9555, replace_existing=True) is True

    def test_port_occupied_by_non_hindsight_returns_false(self):
        """Port occupied by non-hindsight process — returns False."""
        manager = DaemonEmbedManager()
        with (
            patch.object(DaemonEmbedManager, "_is_port_in_use", return_value=True),
            patch("httpx.Client") as mock_httpx_cls,
        ):
            mock_client = MagicMock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.side_effect = httpx.ConnectError("Connection refused")
            mock_httpx_cls.return_value = mock_client

            assert manager._clear_port(9555) is False

    def test_port_occupied_health_non_200_returns_false(self):
        """Port responds but not with 200 — treated as non-hindsight."""
        manager = DaemonEmbedManager()
        with (
            patch.object(DaemonEmbedManager, "_is_port_in_use", return_value=True),
            patch("httpx.Client") as mock_httpx_cls,
        ):
            mock_client = MagicMock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = Mock(status_code=404)
            mock_httpx_cls.return_value = mock_client

            assert manager._clear_port(9555) is False

    def test_pid_not_found_returns_false(self):
        """Hindsight daemon on port but can't find PID — returns False."""
        manager = DaemonEmbedManager()
        with (
            patch.object(DaemonEmbedManager, "_is_port_in_use", return_value=True),
            patch("httpx.Client") as mock_httpx_cls,
            patch.object(DaemonEmbedManager, "_find_pid_on_port", return_value=None),
        ):
            mock_client = MagicMock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = Mock(status_code=200)
            mock_httpx_cls.return_value = mock_client

            assert manager._clear_port(9555, replace_existing=True) is False

    def test_kill_fails_returns_false(self):
        """Hindsight daemon found but won't die — returns False."""
        manager = DaemonEmbedManager()
        with (
            patch.object(DaemonEmbedManager, "_is_port_in_use", return_value=True),
            patch("httpx.Client") as mock_httpx_cls,
            patch.object(DaemonEmbedManager, "_find_pid_on_port", return_value=12345),
            patch.object(DaemonEmbedManager, "_kill_process", return_value=False),
        ):
            mock_client = MagicMock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=False)
            mock_client.get.return_value = Mock(status_code=200)
            mock_httpx_cls.return_value = mock_client

            assert manager._clear_port(9555, replace_existing=True) is False


class TestEnsureDaemonRunning:
    """Tests for daemon_client.ensure_daemon_running."""

    def test_forwards_replace_existing_to_manager(self, config):
        """Wrapper passes replace_existing through to the singleton manager."""
        with patch.object(daemon_client, "_manager") as mock_manager:
            mock_manager.ensure_running.return_value = True

            assert daemon_client.ensure_daemon_running(config, "test-profile", replace_existing=True) is True

            mock_manager.ensure_running.assert_called_once_with(
                config,
                "test-profile",
                extra_args=None,
                replace_existing=True,
            )


class TestDaemonCommand:
    """Tests for explicit daemon command behaviors."""

    def test_do_daemon_start_replace_restarts_even_if_healthy(self, config):
        """`daemon start --replace` must bypass the early already-running return."""
        from hindsight_embed.cli import do_daemon

        args = Namespace(profile="test-profile", daemon_command="start", ui=False, replace=True)
        mock_paths = Mock(log=Path("/tmp/hindsight-daemon.log"), port=9555)

        with (
            patch("hindsight_embed.profile_manager.ProfileManager.resolve_profile_paths", return_value=mock_paths),
            patch.object(daemon_client, "is_daemon_running", return_value=True),
            patch.object(daemon_client, "ensure_daemon_running", return_value=True) as mock_ensure,
        ):
            assert do_daemon(args, config, Mock()) == 0
            mock_ensure.assert_called_once_with(config, "test-profile", replace_existing=True)
