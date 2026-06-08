from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 GitHub 子工程中的主线实验与图表生成流程。"
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="跳过传统机器学习基线实验。",
    )
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="跳过深度学习基线实验。",
    )
    parser.add_argument(
        "--skip-final-model",
        action="store_true",
        help="跳过最终 EEGNet 模型对比实验。",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="跳过论文图表生成。",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="跳过 t-SNE 图生成。",
    )
    return parser.parse_args()


def run_step(title: str, command: list[str]) -> None:
    print(f"\n[开始] {title}")
    print("命令:", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"[完成] {title}")


def main() -> None:
    args = parse_args()
    python = sys.executable

    if not args.skip_baselines:
        run_step(
            "传统机器学习基线实验",
            [python, "scripts/run_gas_baselines.py"],
        )

    if not args.skip_deep:
        run_step(
            "深度学习基线实验",
            [
                python,
                "scripts/run_deep_gas_models.py",
                "--output-dir",
                "results/deep_baselines",
            ],
        )

    if not args.skip_final_model:
        run_step(
            "最终 EEGNet 模型对比实验",
            [
                python,
                "scripts/run_eegnet_variant_suite.py",
                "--experiments",
                "baseline",
                "eca_wide_12_24",
                "--output-dir",
                "results/final_model",
                "--seed",
                "42",
            ],
        )

    if not args.skip_figures:
        run_step(
            "论文图表生成",
            [python, "scripts/generate_publication_figures.py"],
        )

    if not args.skip_tsne:
        run_step(
            "t-SNE 图生成",
            [python, "scripts/generate_tsne_comparison_figure.py"],
        )


if __name__ == "__main__":
    main()
