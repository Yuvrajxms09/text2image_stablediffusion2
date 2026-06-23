from contextlib import contextmanager
from pathlib import Path
import importlib.util
import sys
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _FastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def add_middleware(self, *args, **kwargs):
        pass


class _NextcloudApp:
    def __init__(self, *args, **kwargs):
        self.enabled_state = False
        self.app_cfg = types.SimpleNamespace(app_name="text2image_stablediffusion2")

    def log(self, *args, **kwargs):
        return None


class _AppAPIAuthMiddleware:
    def __init__(self, *args, **kwargs):
        pass


class _ShapeDescriptor:
    def __init__(self, *args, **kwargs):
        pass


class _TaskProcessingProvider:
    def __init__(self, *args, **kwargs):
        pass


def _build_import_stubs():
    return {
        "niquests": _module(
            "niquests",
            codes=types.SimpleNamespace(too_many_requests=429),
            exceptions=_module(
                "niquests.exceptions",
                ConnectionError=ConnectionError,
                Timeout=type("Timeout", (Exception,), {}),
                RequestException=Exception,
            ),
        ),
        "niquests.exceptions": _module(
            "niquests.exceptions",
            ConnectionError=ConnectionError,
            Timeout=type("Timeout", (Exception,), {}),
            RequestException=Exception,
        ),
        "torch": _module("torch", float16="float16", float32="float32"),
        "diffusers": _module("diffusers", AutoPipelineForText2Image=object),
        "fastapi": _module("fastapi", FastAPI=_FastAPI),
        "nc_py_api": _module("nc_py_api", NextcloudApp=_NextcloudApp, NextcloudException=Exception),
        "nc_py_api.ex_app": _module(
            "nc_py_api.ex_app",
            AppAPIAuthMiddleware=_AppAPIAuthMiddleware,
            LogLvl=types.SimpleNamespace(INFO=1, WARNING=2, ERROR=3),
            get_computation_device=lambda: "cpu",
            run_app=lambda *args, **kwargs: None,
            set_handlers=lambda *args, **kwargs: None,
        ),
        "nc_py_api.ex_app.providers.task_processing": _module(
            "nc_py_api.ex_app.providers.task_processing",
            ShapeDescriptor=_ShapeDescriptor,
            ShapeType=types.SimpleNamespace(TEXT="text"),
            TaskProcessingProvider=_TaskProcessingProvider,
        ),
        "PIL": _module(
            "PIL",
            Image=_module("PIL.Image", Image=object),
            ImageDraw=types.SimpleNamespace(Draw=lambda image: None),
            ImageFont=types.SimpleNamespace(load_default=lambda: None),
            PngImagePlugin=types.SimpleNamespace(PngInfo=lambda: None),
        ),
        "PIL.Image": _module("PIL.Image", Image=object),
    }


@contextmanager
def _patched_imports():
    with patch.dict(sys.modules, _build_import_stubs(), clear=False):
        yield


def _load_main():
    with _patched_imports():
        spec = importlib.util.spec_from_file_location("exapp_main", ROOT / "ex_app" / "lib" / "main.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


main = _load_main()


class _FakeCompelForSDXL:
    def __init__(self, pipe, device):
        self.pipe = pipe
        self.device = device


class InitSdxlCompelTests(unittest.TestCase):
    def setUp(self):
        self.original_compel = main.CompelForSDXL
        self.addCleanup(setattr, main, "CompelForSDXL", self.original_compel)

    def test_returns_compel_instance_when_available(self):
        pipe = object()
        main.CompelForSDXL = _FakeCompelForSDXL

        compel = main.init_sdxl_compel(pipe=pipe, device="cuda")

        self.assertIsInstance(compel, _FakeCompelForSDXL)
        self.assertIs(compel.pipe, pipe)
        self.assertEqual(compel.device, "cuda")

    def test_returns_none_when_compel_is_unavailable(self):
        main.CompelForSDXL = None

        compel = main.init_sdxl_compel(pipe=object(), device="cpu")

        self.assertIsNone(compel)
