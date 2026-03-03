"""Basic import smoke tests."""


def test_import_package():
    import webgpu_alpha_hint

    assert hasattr(webgpu_alpha_hint, "__version__")
    assert webgpu_alpha_hint.__version__ == "0.1.0"


def test_import_public_api():
    from webgpu_alpha_hint import console, load_wgsl, log, process_video

    assert callable(process_video)
    assert callable(load_wgsl)
    assert console is not None
    assert log is not None


def test_load_wgsl_returns_string():
    from webgpu_alpha_hint import load_wgsl

    source = load_wgsl("alpha_hint")
    assert isinstance(source, str)
    assert "fn main" in source
