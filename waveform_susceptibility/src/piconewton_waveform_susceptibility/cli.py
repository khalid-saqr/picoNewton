from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import AnalysisConfig
from .figures import create_figures
from .public_analysis import run_analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute waveform susceptibility in anisotropic Womersley flow."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radial-order", type=int, default=150)
    parser.add_argument("--time-points", type=int, default=2048)
    parser.add_argument("--quadrature-nodes", type=int, default=256)
    parser.add_argument("--validation-epsilon", type=float, default=0.08)
    parser.add_argument("--figure-dpi", type=int, default=300)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AnalysisConfig(
        radial_order=args.radial_order,
        time_points=args.time_points,
        quadrature_nodes=args.quadrature_nodes,
        validation_epsilon=args.validation_epsilon,
    )
    result = run_analysis(args.output, config)
    figures = create_figures(args.output, args.figure_dpi)
    print(
        json.dumps(
            {
                "output_root": result["output_root"],
                "figures": [str(path) for path in figures],
                "summary": result["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
