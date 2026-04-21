"""Project config: the single source of truth for paths and tunables.

`Project.save()` / `Project.load()` round-trip through `<project>/config.json`,
so the GUI and CLI share state without a database.

`ProjectsRoot` manages the directory that *contains* projects; the GUI uses it
to list / create / open projects.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

CONFIG_FILENAME = "config.json"
DEFAULT_PROJECTS_ROOT = Path(os.path.expanduser("~/Apps/image_trainer/projects"))

VramProfile = Literal["10gb", "16gb", "24gb"]
MixedPrecision = Literal["no", "fp16", "bf16"]
LRScheduler = Literal["constant", "cosine", "cosine_with_restarts", "linear"]


@dataclass
class Project:
    """Everything about a single LoRA training run lives under `root`."""

    root: Path

    # identity / subject
    trigger_word: str = "ohwx person"

    # data prep
    target_size: int = 1024
    caption_model_id: str = "Salesforce/blip-image-captioning-large"

    # training - what to train on
    base_model_path: Optional[Path] = None
    resolution: int = 1024

    # training - LoRA shape
    lora_rank: int = 32
    lora_alpha: int = 32
    train_text_encoder: bool = False  # toggled in GUI; off by default to save VRAM

    # training - loss / optimizer / schedule (quality)
    learning_rate: float = 1e-4
    lr_scheduler: LRScheduler = "cosine"
    lr_warmup_steps: int = 50
    min_snr_gamma: float = 5.0  # quality: down-weight easy timesteps
    offset_noise: float = 0.05  # quality: improves contrast / dark scenes

    # training - memory
    mixed_precision: MixedPrecision = "fp16"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    use_xformers: bool = True
    use_8bit_optim: bool = True

    # training - length / checkpointing
    max_train_steps: int = 1500
    checkpointing_steps: int = 100
    validation_steps: int = 200  # 0 disables validation previews
    validation_prompt: str = ""  # falls back to f"{trigger_word}, portrait, studio lighting"
    seed: int = 42

    # runtime
    vram_profile: VramProfile = "10gb"

    # ---------- derived paths ----------

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / "checkpoints"

    @property
    def lora_dir(self) -> Path:
        return self.root / "lora"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def validation_dir(self) -> Path:
        return self.logs_dir / "validation"

    @property
    def config_path(self) -> Path:
        return self.root / CONFIG_FILENAME

    def effective_validation_prompt(self) -> str:
        return self.validation_prompt or f"{self.trigger_word}, portrait, studio lighting"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_dir,
            self.processed_dir,
            self.cache_dir,
            self.checkpoints_dir,
            self.lora_dir,
            self.outputs_dir,
            self.logs_dir,
            self.validation_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # ---------- persistence ----------

    def save(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["root"] = str(self.root)
        data["base_model_path"] = (
            str(self.base_model_path) if self.base_model_path is not None else None
        )
        self.config_path.write_text(json.dumps(data, indent=2))
        return self.config_path

    @classmethod
    def load(cls, project_dir: Path) -> "Project":
        project_dir = Path(project_dir)
        path = project_dir / CONFIG_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No {CONFIG_FILENAME} at {project_dir!s}. Run `trainer init` first."
            )
        raw = json.loads(path.read_text())
        raw["root"] = Path(raw["root"])
        raw["base_model_path"] = Path(raw["base_model_path"]) if raw.get("base_model_path") else None
        # Forward-compat: drop unknown keys, supply defaults for missing keys.
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in raw.items() if k in allowed}
        return cls(**clean)


@dataclass
class ProjectsRoot:
    """Directory that contains many project directories.

    The GUI uses this for the project browser (list / create / open)."""

    root: Path = DEFAULT_PROJECTS_ROOT

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[Path]:
        if not self.root.exists():
            return []
        out: list[Path] = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and (p / CONFIG_FILENAME).exists():
                out.append(p)
        return out

    def create(self, name: str, **overrides) -> Project:
        self.ensure()
        dst = self.root / name
        if dst.exists():
            raise FileExistsError(f"Project already exists: {dst}")
        project = Project(root=dst, **overrides)
        project.ensure_dirs()
        project.save()
        return project
