from pathlib import Path
import yaml


def test_frozen_protocol():
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load((root / "configs/protocol.yaml").read_text(encoding="utf-8"))
    assert protocol["protocol"]["status"] == "scientific_scope_frozen"
    assert len(protocol["hypotheses"]) == 7
    assert len(protocol["arteries"]) == 6
