from pathlib import Path


def test_public_tree_excludes_internal_workflow_vocabulary():
    root = Path(__file__).parents[1]
    forbidden = ("st" + "ep", "ga" + "te")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {
            ".py",
            ".md",
            ".toml",
            ".ipynb",
            ".yml",
        }:
            continue
        text = path.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert word not in text, f"{word!r} found in {path.relative_to(root)}"
