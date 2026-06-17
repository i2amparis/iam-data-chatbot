import unittest
from dataclasses import dataclass

from quality_gate import build_commands, run_commands


@dataclass
class _Completed:
    returncode: int


class QualityGateTests(unittest.TestCase):
    def test_build_commands_includes_live_gates_when_live_url_is_set(self):
        commands = build_commands(live_url="http://127.0.0.1:8000/query")
        names = [command.name for command in commands]

        self.assertIn("unit tests", names)
        self.assertIn("main live eval", names)
        self.assertIn("conversation live eval", names)
        self.assertIn("frontend response audit", names)

    def test_build_commands_can_include_link_validation(self):
        commands = build_commands(include_link_validation=True, include_static_eval=False)

        self.assertEqual(commands[-1].name, "IAM PARIS link validation")

    def test_run_commands_returns_zero_when_all_pass(self):
        calls = []

        def runner(args):
            calls.append(args)
            return _Completed(0)

        code = run_commands(build_commands(include_static_eval=False), runner=runner)

        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)

    def test_run_commands_stops_on_required_failure(self):
        calls = []

        def runner(args):
            calls.append(args)
            return _Completed(1)

        code = run_commands(build_commands(include_static_eval=True), runner=runner)

        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
