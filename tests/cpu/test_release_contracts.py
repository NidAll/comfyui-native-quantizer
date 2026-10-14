"""Release identity and opt-in integration contracts."""
import ast
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.support import comfyui_root, isolated_module as setUpModule

ROOT = Path(__file__).resolve().parents[2]


class ReleaseContracts(unittest.TestCase):
    def test_distribution_and_runtime_versions_agree(self):
        from native_quantizer.identity import get_converter_version
        core = (ROOT / 'pyproject.toml').read_text()
        self.assertEqual(re.search(r'^version = "([^"]+)"', core, re.M)[1], get_converter_version())
        self.assertNotIn('textual', core)

    def test_missing_requested_consumer_is_an_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(unittest.SkipTest):
                comfyui_root()
            os.environ['COMFYUI_REQUIRED'] = '1'
            with self.assertRaisesRegex(RuntimeError, 'requires COMFYUI_ROOT'):
                comfyui_root()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'COMFYUI_ROOT': directory}):
                with self.assertRaisesRegex(RuntimeError, 'not a ComfyUI'):
                    comfyui_root()

    def test_python_sources_parse_under_minimum_supported_grammar(self):
        for folder in ('native_quantizer', 'scripts'):
            for path in (ROOT / folder).rglob('*.py'):
                with self.subTest(path=path.relative_to(ROOT)):
                    ast.parse(path.read_text(), feature_version=(3, 10))


if __name__ == '__main__':
    unittest.main()
