from __future__ import annotations

import json
import os
import sys
import types
import unittest
from unittest.mock import patch

psycopg_module = types.ModuleType("psycopg")
psycopg_sql_module = types.ModuleType("psycopg.sql")
psycopg_conninfo_module = types.ModuleType("psycopg.conninfo")
psycopg_rows_module = types.ModuleType("psycopg.rows")
psycopg_types_module = types.ModuleType("psycopg.types")
psycopg_json_module = types.ModuleType("psycopg.types.json")


class _Connection:
    def __class_getitem__(cls, item):
        return cls


class _OperationalError(Exception):
    pass


psycopg_module.Connection = _Connection
psycopg_module.OperationalError = _OperationalError
psycopg_module.connect = lambda *args, **kwargs: None
psycopg_sql_module.SQL = lambda value: value
psycopg_sql_module.Identifier = lambda value: value
psycopg_module.sql = psycopg_sql_module
psycopg_conninfo_module.conninfo_to_dict = lambda value: {}
psycopg_conninfo_module.make_conninfo = lambda value="", **kwargs: value
psycopg_rows_module.dict_row = object()
psycopg_json_module.Jsonb = lambda value: value
sys.modules.setdefault("psycopg", psycopg_module)
sys.modules.setdefault("psycopg.sql", psycopg_sql_module)
sys.modules.setdefault("psycopg.conninfo", psycopg_conninfo_module)
sys.modules.setdefault("psycopg.rows", psycopg_rows_module)
sys.modules.setdefault("psycopg.types", psycopg_types_module)
sys.modules.setdefault("psycopg.types.json", psycopg_json_module)

fastapi_module = types.ModuleType("fastapi")
fastapi_encoders_module = types.ModuleType("fastapi.encoders")
fastapi_middleware_module = types.ModuleType("fastapi.middleware")
fastapi_cors_module = types.ModuleType("fastapi.middleware.cors")
fastapi_responses_module = types.ModuleType("fastapi.responses")
pydantic_module = types.ModuleType("pydantic")


class _FastAPI:
    def __init__(self, *args, **kwargs):
        pass

    def add_middleware(self, *args, **kwargs):
        return None

    def get(self, *args, **kwargs):
        return lambda func: func

    def post(self, *args, **kwargs):
        return lambda func: func

    def delete(self, *args, **kwargs):
        return lambda func: func

    def on_event(self, *args, **kwargs):
        return lambda func: func


class _HTTPException(Exception):
    def __init__(self, status_code: int, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class _BaseModel:
    pass


fastapi_module.FastAPI = _FastAPI
fastapi_module.HTTPException = _HTTPException
fastapi_module.Query = lambda default=None, **kwargs: default
fastapi_encoders_module.jsonable_encoder = lambda value: value
fastapi_cors_module.CORSMiddleware = object
fastapi_responses_module.JSONResponse = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
pydantic_module.BaseModel = _BaseModel
pydantic_module.Field = lambda default=None, **kwargs: default
sys.modules.setdefault("fastapi", fastapi_module)
sys.modules.setdefault("fastapi.encoders", fastapi_encoders_module)
sys.modules.setdefault("fastapi.middleware", fastapi_middleware_module)
sys.modules.setdefault("fastapi.middleware.cors", fastapi_cors_module)
sys.modules.setdefault("fastapi.responses", fastapi_responses_module)
sys.modules.setdefault("pydantic", pydantic_module)

from backend.app import experiment_runner
from backend.app import main


class _FakeResponse:
    status = 200

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "id": "chatcmpl-test",
                "model": "gpt-oss-120b",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "tasks": [
                                        {
                                            "id": "sample-1",
                                            "optimized_query": "Ada Lovelace",
                                        }
                                    ]
                                }
                            )
                        }
                    }
                ],
            }
        ).encode("utf-8")


class _FakePlainResponse:
    status = 200

    def __enter__(self) -> "_FakePlainResponse":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(
            {
                "id": "chatcmpl-test",
                "model": "gpt-oss-120b",
                "choices": [{"message": {"content": "It works."}}],
            }
        ).encode("utf-8")


class CerebrasProviderTest(unittest.TestCase):
    def test_cerebras_provider_uses_direct_endpoint_and_key(self) -> None:
        sample = {
            "sample_id": "sample-1",
            "mention_text": "Ada Lovelace",
            "lookup_text": "Ada Lovelace",
            "dataset": "test",
            "table_id": "table",
            "row_id": 0,
            "col_id": 0,
        }
        config = experiment_runner.normalize_experiment_config(
            {
                "llm_provider": "cerebras",
                "llm_api_key": "",
                "llm_max_retries": 1,
                "llm_max_tokens": 32,
            }
        )

        with patch.dict(os.environ, {"CEREBRAS_API_KEY": "cerebras-test-key"}, clear=False):
            with patch("backend.app.experiment_runner.urlopen", return_value=_FakeResponse()) as urlopen_mock:
                plans, log = experiment_runner._call_llm_batch([sample], config, "batch-1")

        request = urlopen_mock.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))

        self.assertEqual(request.full_url, "https://api.cerebras.ai/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer cerebras-test-key")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(request.get_header("User-agent"), "coverage-dashboard/1.0")
        self.assertEqual(body["model"], "gpt-oss-120b")
        self.assertEqual(body["max_completion_tokens"], 32)
        self.assertEqual(body["reasoning_effort"], "high")
        self.assertNotIn("provider", body)
        self.assertNotIn("include_reasoning", body)
        self.assertEqual(plans["sample-1"]["optimized_query"], "Ada Lovelace")
        self.assertEqual(log["provider"], "cerebras")

    def test_openrouter_route_to_cerebras_uses_openrouter_key(self) -> None:
        sample = {
            "sample_id": "sample-1",
            "mention_text": "Ada Lovelace",
            "lookup_text": "Ada Lovelace",
            "dataset": "test",
            "table_id": "table",
            "row_id": 0,
            "col_id": 0,
        }
        config = experiment_runner.normalize_experiment_config(
            {
                "llm_provider": "openrouter",
                "llm_provider_name": "Cerebras",
                "llm_api_key": "",
                "openrouter_api_key": "",
                "cerebras_api_key": "",
                "llm_max_retries": 1,
            }
        )

        with patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "openrouter-test-key", "CEREBRAS_API_KEY": "cerebras-test-key"},
            clear=False,
        ):
            with patch("backend.app.experiment_runner.urlopen", return_value=_FakeResponse()) as urlopen_mock:
                experiment_runner._call_llm_batch([sample], config, "batch-1")

        request = urlopen_mock.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))

        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer openrouter-test-key")
        self.assertEqual(request.get_header("Accept"), "application/json")
        self.assertEqual(request.get_header("User-agent"), "coverage-dashboard/1.0")
        self.assertEqual(body["provider"]["order"], ["cerebras"])

    def test_settings_smoke_call_does_not_require_json_output(self) -> None:
        config = experiment_runner.normalize_experiment_config(
            {
                "llm_provider": "cerebras",
                "llm_api_key": "",
                "llm_max_tokens": 16,
            }
        )

        with patch.dict(os.environ, {"CEREBRAS_API_KEY": "cerebras-test-key"}, clear=False):
            with patch("backend.app.main.urlopen", return_value=_FakePlainResponse()) as urlopen_mock:
                content, metadata = main._llm_plain_request([{"role": "user", "content": "Reply briefly."}], config)

        request = urlopen_mock.call_args.args[0]
        body = json.loads(request.data.decode("utf-8"))

        self.assertEqual(content, "It works.")
        self.assertEqual(metadata["response_model"], "gpt-oss-120b")
        self.assertNotIn("response_format", body)
        self.assertNotIn("reasoning", body)
        self.assertNotIn("reasoning_effort", body)


if __name__ == "__main__":
    unittest.main()
