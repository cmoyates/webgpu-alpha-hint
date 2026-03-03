"""Import smoke tests and public API validation."""

import logging


def test_import_package():
    import webgpu_alpha_hint

    assert hasattr(webgpu_alpha_hint, "__version__")
    assert webgpu_alpha_hint.__version__ == "0.1.0"


def test_import_public_api():
    from webgpu_alpha_hint import (
        create_texture,
        load_wgsl,
        process_video,
        readback_r_channel,
        upload_rgba,
    )

    assert callable(process_video)
    assert callable(load_wgsl)
    assert callable(create_texture)
    assert callable(upload_rgba)
    assert callable(readback_r_channel)


def test_load_wgsl_returns_string():
    from webgpu_alpha_hint import load_wgsl

    source = load_wgsl("alpha_hint")
    assert isinstance(source, str)
    assert "fn main" in source


def test_all_shaders_loadable():
    """Every bundled shader should load without error."""
    from webgpu_alpha_hint import load_wgsl

    for name in ("alpha_hint", "blur", "morphology"):
        source = load_wgsl(name)
        assert isinstance(source, str)
        assert len(source) > 0


def test_import_does_not_hijack_logging():
    """Importing the package must not call logging.basicConfig or add handlers."""
    root = logging.getLogger()
    handlers_before = list(root.handlers)

    assert root.handlers == handlers_before, "Import should not modify root logger handlers"


def test_console_module_accessible():
    """Console/log are still importable from the submodule for advanced use."""
    from webgpu_alpha_hint.console import console, log

    assert console is not None
    assert log is not None


def test_cli_entry_importable():
    """CLI module should be importable without side effects."""
    from webgpu_alpha_hint._cli import cli

    assert callable(cli)
