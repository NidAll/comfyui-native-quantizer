"""Failure-path checks: telemetry must never modify source or control inputs."""
import hashlib
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from native_quantizer.cli.entrypoint import entrypoint
from native_quantizer.errors import OutputError, UsageError
from native_quantizer import api, events
from native_quantizer.options import ConversionOptions
from tests_support import isolated_module as setUpModule


class CliSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'input.safetensors'
        save_file({'double_blocks.0.img_attn.qkv.weight': torch.ones(64, 256),
                   'img_in.weight': torch.ones(64, 64)}, str(self.source))

    def digest(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_hardlink_event_alias_is_rejected_before_truncation(self):
        alias = self.root / 'events.jsonl'
        os.link(self.source, alias)
        before = self.digest(self.source)
        self.assertEqual(entrypoint([str(self.source), '--no-config', '--events-jsonl', str(alias),
                                     '--inspect', '--json']), 2)
        self.assertEqual(self.digest(self.source), before)

    def test_index_and_directory_shards_are_protected_in_cli_and_api(self):
        index = self.root / 'model.safetensors.index.json'
        index.write_text(json.dumps({'weight_map': {
            'double_blocks.0.img_attn.qkv.weight': self.source.name,
            'img_in.weight': self.source.name}}))
        before = self.digest(self.source)
        for model in (str(index), str(self.root)):
            with self.subTest(model=model):
                self.assertEqual(entrypoint([model, '--no-config', '--events-jsonl', str(self.source),
                                            '--inspect', '--json']), 1)
                with self.assertRaises((OutputError, UsageError)):
                    api.convert(ConversionOptions(model=model, events_jsonl=str(self.source),
                                                  output=str(self.root / 'output.safetensors'),
                                                  format='int8', device='cpu'))
                self.assertEqual(self.digest(self.source), before)

    def test_event_stream_cannot_alias_output_journal_or_index(self):
        output = self.root / 'output.safetensors'
        state = Path(str(output) + '.state.json')
        state.write_text('preserve existing state')
        before = self.digest(state)
        self.assertEqual(entrypoint([str(self.source), '--no-config', '--output', str(output),
                                     '--format', 'int8', '--device', 'cpu',
                                     '--events-jsonl', str(state)]), 1)
        self.assertEqual(self.digest(state), before)
        self.assertFalse(output.exists())

    def test_control_file_collisions_fail_without_mutating_inputs(self):
        config = self.root / 'defaults.json'
        config.write_text(json.dumps({'schema': 1, 'device': 'cpu'}))
        before = self.digest(config)
        for flag in ('--report', '--json-log', '--events-jsonl', '--output'):
            with self.subTest(flag=flag):
                self.assertEqual(entrypoint([str(self.source), '--config', str(config),
                                             flag, str(config), '--inspect']), 2)
                self.assertEqual(self.digest(config), before)

    def test_uncreated_artifacts_through_symlinked_directories_alias(self):
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        from native_quantizer.files import _same_path
        self.assertTrue(_same_path(str(alias / 'future.json'), str(self.root / 'future.json')))

    def test_typed_api_rejects_invalid_device_before_telemetry(self):
        sink = self.root / 'events.jsonl'
        sink.write_text('keep telemetry')
        before = self.digest(sink)
        with self.assertRaisesRegex(UsageError, 'device'):
            api.convert(ConversionOptions(model=str(self.source), device='typo', events_jsonl=str(sink)))
        self.assertEqual(self.digest(sink), before)

    def test_self_test_fixtures_are_released_before_the_next_case(self):
        from native_quantizer.testing import self_tests
        created = []

        def failing_case():
            path = Path(self_tests._tmpdir())
            created.append(path)
            (path / 'fixture').write_bytes(b'synthetic')
            raise RuntimeError('intentional fixture failure')

        def next_case():
            self.assertFalse(created[0].exists())
            return 'previous scratch removed'

        with patch.object(self_tests, 'SELF_TEST_CASES', [('failed', failing_case), ('next', next_case)]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self_tests.run_self_tests(), 1)
        self.assertFalse(created[0].exists())

    def test_invalid_request_does_not_open_or_reuse_an_event_sink(self):
        sink = self.root / 'events.jsonl'
        sink.write_text('keep telemetry')
        before = self.digest(sink)
        self.assertEqual(entrypoint(['--no-config', '--list-formats', '--diagnose',
                                     '--events-jsonl', str(sink)]), 2)
        self.assertEqual(self.digest(sink), before)
        self.assertFalse(events.enabled())
        with self.assertRaises(UsageError):
            api.convert(ConversionOptions(model=str(self.source), format='invalid', events_jsonl=str(sink)))
        self.assertEqual(self.digest(sink), before)
        events.configure(str(sink))
        sink.write_text('previous invocation')
        before = self.digest(sink)
        self.assertEqual(entrypoint(['--no-config', '--json']), 2)
        self.assertEqual(self.digest(sink), before)
        self.assertFalse(events.enabled())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            entrypoint(['--no-config', '--events-jsonl', str(sink)])
        self.assertEqual(exc.exception.code, 2)
        self.assertEqual(self.digest(sink), before)


if __name__ == '__main__':
    unittest.main()
