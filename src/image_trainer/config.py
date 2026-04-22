"""Project config: the single source of truth for paths and tunables.

`Project.save()` / `Project.load()` round-trip through `<project>/config.json`,
so the GUI and CLI share state without a database.

`ProjectsRoot` manages the directory that *contains* projects; the GUI uses it
to list / create / open projects.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
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
    #: When True, :func:`pipeline.resize.resize_dataset` detects the largest
    #: face in each source and places it on a rule-of-thirds intersection
    #: (using the face's natural quadrant in the source to choose which
    #: intersection). Images where no face is detected are still written
    #: with a centre-crop fallback but are marked ``include=False`` in
    #: review.json so you can eyeball them. Requires the optional extra:
    #: ``pip install -e ".[face]"``. Falls back gracefully when missing.
    face_aware_crop: bool = True
    caption_model_id: str = "Salesforce/blip-image-captioning-large"
    #: Which captioner(s) to run in step 3.
    #: - ``"blip"``:   BLIP sentence only (classic, safer). Fast.
    #: - ``"wd14"``:   WD14 Danbooru tags only (NSFW-aware, no sentence context).
    #: - ``"both"``:   BLIP sentence + WD14 tags concatenated. Recommended for
    #:                 NSFW person LoRAs where body/anatomy detail matters.
    captioner: str = "both"
    #: WD14 model used when ``captioner`` is ``"wd14"`` or ``"both"``. Must be
    #: a SmilingWolf ONNX-format WD14 repo on Hugging Face.
    wd14_model_id: str = "SmilingWolf/wd-v1-4-moat-tagger-v2"

    # training - what to train on
    base_model_path: Optional[Path] = None
    resolution: int = 1024

    # training - LoRA shape
    lora_rank: int = 32
    lora_alpha: int = 32
    train_text_encoder: bool = False  # toggled in GUI; off by default to save VRAM
    # TE LoRA-specific knobs. Only meaningful when train_text_encoder=True.
    # Kept separate from UNet LoRA because TEs use less capacity (so lower
    # rank) and are more sensitive to learning rate (so lower LR).
    te_lora_rank: int = 8
    te_lora_alpha: int = 8
    te_learning_rate: float = 5e-5
    # Gradient checkpointing on both text encoders. Required to fit TE LoRA
    # on a 10 GB card; costs ~10% step time. Disable on 16 GB+ for a speed
    # bump.
    te_gradient_checkpointing: bool = True

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

    # generation defaults (Review + Generate tabs)
    default_negative_prompt: str = (
        "low quality, blurry, deformed, extra fingers, bad anatomy, watermark"
    )
    # Quick-insert tags the Review tab chips bar exposes. Click a chip in the
    # GUI and the token is appended to the current image's caption. Edit this
    # list per project to taste — it lives in config.json so each project
    # can have its own chip palette (e.g. NSFW-specific chips for adult
    # datasets, style chips for illustration datasets, etc).
    prompt_chips: list = field(
        default_factory=lambda: [
            # framing
            "closeup",
            "half body",
            "full body",
            "portrait",
            "three quarter view",
            "back view",
            "side view",
            "front view",
            # lighting
            "natural light",
            "studio light",
            "soft light",
            "harsh light",
            "golden hour",
            "low light",
            "backlit",
            # setting
            "outdoor",
            "indoor",
            "bedroom",
            "bathroom",
            "beach",
            "park",
            "studio",
            # pose / action
            "standing",
            "sitting",
            "kneeling",
            "lying down",
            "on back",
            "on side",
            "bent over",
            "arms raised",
            "legs spread",
            "legs crossed",
            # expression
            "smiling",
            "serious",
            "looking at camera",
            "looking away",
            "eyes closed",
            "mouth open",
            # clothing (SFW)
            "dress",
            "jeans",
            "t-shirt",
            "sweater",
            "jacket",
            "skirt",
            "swimsuit",
            "bikini",
            # clothing (NSFW / implicit)
            "lingerie",
            "bra and panties",
            "stockings",
            "fishnets",
            "topless",
            "nude",
            "wet",
            # explicit details (adult LoRAs)
            "nipples visible",
            "cleavage",
            "bare shoulders",
            "bare legs",
            "spread legs",
            # camera
            "35mm photo",
            "film grain",
            "shallow depth of field",
            "wide angle",
        ]
    )

    # runtime
    # Reserved for future profile-aware defaults. Currently not read by any
    # training/generation code path — safe to ignore or delete from config.json.
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
        """Prompt used for in-training validation previews.

        Returns :attr:`validation_prompt` if the user has set one, otherwise a
        sensible default constructed from the trigger word.
        """
        return self.validation_prompt or f"{self.trigger_word}, portrait, studio lighting"

    def ensure_dirs(self) -> None:
        """Create every per-project subdirectory this code might write to.

        Idempotent. Called at the top of each pipeline step so users can
        invoke any step in any order without hitting "directory not found".
        """
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
        """Write the current project state to ``<root>/config.json``.

        Path fields are stringified for JSON. The file is the single source
        of truth shared between the CLI and GUI; both read it on every
        invocation, so in-memory mutations in the GUI are flushed via
        ``_save_settings`` before dispatching a CLI subprocess.
        """
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
        """Load a project from its ``config.json``.

        Forward-compatible: any keys in ``config.json`` that aren't fields on
        the current :class:`Project` are silently dropped, and missing keys
        fall back to dataclass defaults. That way upgrading the code doesn't
        invalidate an older project directory.

        Raises:
            FileNotFoundError: if the project hasn't been initialized yet.
        """
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
    """A directory that contains many project directories.

    The GUI uses this for its top-bar project browser (list / create / open).
    The CLI resolves bare project names like ``me`` against :data:`root`
    (see :func:`image_trainer.cli._resolve_project_dir`).
    """

    root: Path = DEFAULT_PROJECTS_ROOT

    def ensure(self) -> None:
        """Create the root directory if it doesn't exist yet."""
        self.root.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[Path]:
        """Return every subdirectory of :attr:`root` that contains a
        ``config.json`` — i.e. every initialized project.

        Empty list if :attr:`root` doesn't exist.
        """
        if not self.root.exists():
            return []
        out: list[Path] = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and (p / CONFIG_FILENAME).exists():
                out.append(p)
        return out

    def create(self, name: str, **overrides) -> Project:
        """Scaffold a new project under ``<root>/<name>/``.

        Creates the standard directory layout, writes the initial
        ``config.json``, and returns the loaded :class:`Project`. Extra
        keyword arguments are forwarded to :class:`Project` so callers can
        override fields at creation time (e.g. trigger word, base model).

        Raises:
            FileExistsError: if a project with that name already exists.
        """
        self.ensure()
        dst = self.root / name
        if dst.exists():
            raise FileExistsError(f"Project already exists: {dst}")
        project = Project(root=dst, **overrides)
        project.ensure_dirs()
        project.save()
        return project
