# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the documentation command without installing WarpConvNet."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("mkdocs", reason="Install the docs dependencies to test documentation builds")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_docs.py"
_SPEC = importlib.util.spec_from_file_location("warpconvnet_build_docs_test", _SCRIPT)
build_docs_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_docs_module)


@pytest.fixture
def docs_project(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "docs").mkdir(parents=True)
    (project / "docs" / "index.md").write_text("# Documentation smoke test\n", encoding="utf-8")
    (project / "mkdocs.yml").write_text("site_name: Test documentation\n", encoding="utf-8")
    monkeypatch.setattr(build_docs_module, "__file__", str(project / "scripts" / "build_docs.py"))
    monkeypatch.chdir(tmp_path)
    return project


def test_build_docs_without_generator_scripts(docs_project, capsys):
    site = docs_project / "site"
    site.mkdir()
    stale_page = site / "stale.html"
    stale_page.write_text("outdated page", encoding="utf-8")

    assert build_docs_module.build_docs() == 0

    assert "Documentation smoke test" in (site / "index.html").read_text(encoding="utf-8")
    assert not stale_page.exists()
    assert "Documentation built successfully!" in capsys.readouterr().out


def test_build_docs_reports_mkdocs_failure(docs_project, capsys):
    (docs_project / "mkdocs.yml").write_text(
        "site_name: Test documentation\ndocs_dir: missing-docs\n", encoding="utf-8"
    )

    assert build_docs_module.build_docs() == 1

    captured = capsys.readouterr()
    assert "Error building documentation:" in captured.err
    assert "missing-docs" in captured.err
    assert "Documentation built successfully!" not in captured.out
