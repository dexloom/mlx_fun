"""Tests for CLI command registration and basic validation."""

from click.testing import CliRunner

from mlx_fun.cli import main


def test_cli_group():
    """CLI group is accessible."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "MLX-FUN" in result.output


def test_collect_command_registered():
    """Collect command exists and shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["collect", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--dataset" in result.output
    assert "--output" in result.output


def test_prune_command_registered():
    """Prune command exists and shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["prune", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--saliency" in result.output
    assert "--n-prune" in result.output


def test_smoke_test_command_registered():
    """Smoke-test command exists and shows help."""
    runner = CliRunner()
    result = runner.invoke(main, ["smoke-test", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--prompt" in result.output


def test_collect_requires_args():
    """Collect fails without required arguments."""
    runner = CliRunner()
    result = runner.invoke(main, ["collect"])
    assert result.exit_code != 0
    assert "Missing" in result.output or "required" in result.output.lower()


def test_prune_requires_args():
    """Prune fails without required arguments."""
    runner = CliRunner()
    result = runner.invoke(main, ["prune"])
    assert result.exit_code != 0
