from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import env_report


class EnvReportTests(unittest.TestCase):
    def test_requirement_hash_ignores_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lf, crlf = Path(tmp) / "lf.txt", Path(tmp) / "crlf.txt"
            lf.write_bytes(b"numpy==2.2.6\ntorch==2.13.0\n")
            crlf.write_bytes(b"numpy==2.2.6\r\ntorch==2.13.0\r\n")
            self.assertEqual(
                env_report.sha256_text_normalized(lf),
                env_report.sha256_text_normalized(crlf),
            )

    def test_report_outside_git_repo_has_no_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text("numpy==2.2.6\n", encoding="utf-8")
            with mock.patch.object(env_report, "torch_info", return_value={"version": "x"}):
                report = env_report.build_report(root)
        self.assertEqual(report["python_executable"], "<fora do repositório>")
        self.assertEqual(list(report["requirements_sha256_lf"]), ["requirements.txt"])
        self.assertEqual(report["third_party_repos"], {})
        self.assertIsNone(report["repo"]["commit"])

    def test_provenance_hashes_inputs_and_tolerates_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stats.json").write_text("{}", encoding="utf-8")
            with mock.patch.object(env_report, "torch_info", return_value={"version": "x"}):
                block = env_report.provenance({"stats": root / "stats.json", "absent": root / "no.json", "none": None}, root)
        self.assertEqual(len(block["input_sha256"]["stats"]), 64)
        self.assertIsNone(block["input_sha256"]["absent"])
        self.assertIsNone(block["input_sha256"]["none"])
        self.assertIsNone(block["stconvs2s"])

    def test_missing_package_is_reported_as_none(self) -> None:
        self.assertIsNone(env_report._package_version("pacote-inexistente-wsts"))


if __name__ == "__main__":
    unittest.main()
