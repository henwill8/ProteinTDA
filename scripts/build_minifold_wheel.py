"""Build the patched MiniFold wheel used by ProteinTDA."""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WHEELS_DIR = REPO_ROOT / "wheels"
UPSTREAM = "https://github.com/jwohlwend/minifold.git"
WHEEL_NAME = "minifold-0.1.0-py3-none-any.whl"

PYPROJECT = """\
[build-system]
requires = ["setuptools >= 61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "minifold"
version = "0.1.0"
requires-python = ">=3.9"
description = "MiniFold"
readme = "README.md"
dependencies = [
    "numpy",
    "scipy",
    "pytorch_lightning",
    "torch>=2.3.0",
    "fair-esm",
    "biopython==1.81",
    "edit_distance",
    "pandas",
    "modelcif",
    "dm-tree",
    "psutil",
    "click",
    "ml_collections",
    "einops",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["minifold*"]

[tool.setuptools.package-data]
minifold = ["utils/stereo_chemical_props.txt"]
"""


def _optional_triton_imports(text: str) -> str:
    return text.replace(
        "import triton\nimport triton.language as tl",
        "try:\n    import triton\n    import triton.language as tl\nexcept ImportError:\n    triton = None\n    tl = None",
        1,
    )


def _wrap_block(text: str, start_marker: str, end_marker: str, wrapper: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    block = text[start:end]
    indented = "\n".join(f"    {line}" if line else "" for line in block.splitlines())
    return text[:start] + f"{wrapper}\n{indented}\n" + text[end:]


def patch_kernel_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = _optional_triton_imports(text)
    kernel_name = "gating_kernel" if path.name == "gating.py" else "mlp_kernel"
    text = _wrap_block(
        text,
        "@triton.autotune",
        f"\ndef {kernel_name}",
        "if triton is not None:",
    )
    text = text.replace(
        'if X.device.type == "cuda":',
        'if X.device.type == "cuda" and triton is not None:',
        1,
    )
    if "@triton.testing.perf_report" in text:
        text = _wrap_block(
            text,
            "@triton.testing.perf_report",
            "\ndef clear_gradients",
            "if triton is not None:",
        )
        text = text.replace(
            "\ndef clear_gradients",
            "\nelse:\n    benchmark = None\n\ndef clear_gradients",
            1,
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False)",
            "        benchmark.run(print_data=True, show_plots=False) if benchmark is not None else None",
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False, device=\"cuda\")",
            "        benchmark.run(print_data=True, show_plots=False, device=\"cuda\") if benchmark is not None else None",
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False, device=\"mps\")",
            "        benchmark.run(print_data=True, show_plots=False, device=\"mps\") if benchmark is not None else None",
        )
        text = text.replace(
            "        benchmark.run(print_data=True, show_plots=False, device=\"cpu\")",
            "        benchmark.run(print_data=True, show_plots=False, device=\"cpu\") if benchmark is not None else None",
        )
    path.write_text(text, encoding="utf-8")


def patch_heads(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "class TMScoreHead" in text:
        return

    text = text.replace(
        "class AuxiliaryHeads(nn.Module):",
        '''class TMScoreHead(nn.Module):
    def __init__(self, c_z, no_bins, **kwargs):
        super().__init__()
        self.linear = nn.Linear(c_z, no_bins)
        init.final_init_(self.linear.weight)
        init.bias_init_zero_(self.linear.bias)

    def forward(self, z):
        return self.linear(z)


class ExperimentallyResolvedHead(nn.Module):
    def __init__(self, c_s, c_out, **kwargs):
        super().__init__()
        self.linear = nn.Linear(c_s, c_out)
        init.final_init_(self.linear.weight)
        init.bias_init_zero_(self.linear.bias)

    def forward(self, s):
        return self.linear(s)


class AuxiliaryHeads(nn.Module):''',
    )

    text = text.replace(
        """    def __init__(self, config):
        super().__init__()
        self.plddt = PerResidueLDDTCaPredictor(
            **config["lddt"],
        )
        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits
        aux_out["plddt"] = compute_plddt(lddt_logits)
        return aux_out""",
        """    def __init__(self, config):
        super().__init__()
        self.plddt = PerResidueLDDTCaPredictor(**config["lddt"])
        self.experimentally_resolved = ExperimentallyResolvedHead(
            **config["experimentally_resolved"],
        )
        tm_cfg = config["tm"]
        self.tm_enabled = bool(tm_cfg.get("enabled", False))
        if self.tm_enabled:
            self.tm = TMScoreHead(**tm_cfg)
        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits
        aux_out["plddt"] = compute_plddt(lddt_logits)
        aux_out["experimentally_resolved_logits"] = self.experimentally_resolved(
            outputs["single"]
        )
        if self.tm_enabled:
            aux_out["tm_logits"] = self.tm(outputs["pair"])
        return aux_out""",
    )
    path.write_text(text, encoding="utf-8")


def patch_loss(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "self.config.violation.weight != 0.0" in text:
        return

    text = text.replace(
        """        \"\"\"
        if "violation" not in out.keys():
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )
        \"\"\"

        if "renamed_atom14_gt_positions" not in out.keys():""",
        """        if self.config.violation.weight != 0.0 and "violation" not in out:
            out["violation"] = find_structural_violations(
                batch,
                out["sm"]["positions"][-1],
                **self.config.violation,
            )

        if "renamed_atom14_gt_positions" not in out.keys():""",
    )

    text = text.replace(
        """        \"\"\"
        "distogram": lambda: distogram_loss(
            logits=out["distogram_logits"],
            **{**batch, **self.config.distogram},
        ),
        "experimentally_resolved": lambda: experimentally_resolved_loss(
            logits=out["experimentally_resolved_logits"],
            **{**batch, **self.config.experimentally_resolved},
        ),
        \"\"\"

        \"\"\"
        "masked_msa": lambda: masked_msa_loss(
            logits=out["masked_msa_logits"],
            **{**batch, **self.config.masked_msa},
        ),

        "violation": lambda: violation_loss(
            out["violation"],
            **batch,
        ),
        \"\"\"

        if self.config.tm.enabled:
            loss_fns["tm"] = lambda: tm_loss(
                logits=out["tm_logits"],
                **{**batch, **out, **self.config.tm},
            )""",
        """        if self.config.experimentally_resolved.weight != 0.0:
            loss_fns["experimentally_resolved"] = lambda: experimentally_resolved_loss(
                logits=out["experimentally_resolved_logits"],
                **{**batch, **self.config.experimentally_resolved},
            )

        if self.config.violation.weight != 0.0:
            loss_fns["violation"] = lambda: violation_loss(
                out["violation"],
                **batch,
            )

        if self.config.tm.enabled and self.config.tm.weight != 0.0:
            # MiniFold stores frames as 4x4 matrices, but tm_loss expects a
            # quaternion tensor_7, so convert final_affine_tensor here.
            tm_kwargs = {**batch, **out, **self.config.tm}
            tm_kwargs["final_affine_tensor"] = Rigid.from_tensor_4x4(
                out["final_affine_tensor"]
            ).to_tensor_7()
            loss_fns["tm"] = lambda: tm_loss(logits=out["tm_logits"], **tm_kwargs)""",
    )
    path.write_text(text, encoding="utf-8")


def main() -> int:
    WHEELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "minifold"
        subprocess.run(["git", "clone", "--depth", "1", UPSTREAM, str(src)], check=True)
        (src / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
        patch_kernel_file(src / "minifold" / "model" / "kernels" / "gating.py")
        patch_kernel_file(src / "minifold" / "model" / "kernels" / "mlp.py")
        patch_heads(src / "minifold" / "model" / "heads.py")
        patch_loss(src / "minifold" / "train" / "loss.py")
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", str(src), "-w", str(WHEELS_DIR), "--no-deps"],
            check=True,
        )
    built = WHEELS_DIR / WHEEL_NAME
    if not built.exists():
        wheels = list(WHEELS_DIR.glob("minifold-*.whl"))
        if len(wheels) != 1:
            raise SystemExit(f"Expected one wheel in {WHEELS_DIR}, found: {wheels}")
        wheels[0].replace(built)
    print(f"Built {built}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
