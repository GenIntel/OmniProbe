import inspect
from pathlib import Path
from collections.abc import Sequence

from hydra.utils import get_object, instantiate
from omegaconf import OmegaConf


class BackboneContract:
    def __init__(
        self,
        target: str,
        supported_outputs: Sequence[str],
        default_global_output: str | None,
        supports_multilayer: bool,
        supports_layer_selection: bool,
    ) -> None:
        self.target = target
        self.supported_outputs = tuple(supported_outputs)
        self.default_global_output = default_global_output
        self.supports_multilayer = supports_multilayer
        self.supports_layer_selection = supports_layer_selection

    def supports_output(self, output_name: str) -> bool:
        return output_name in self.supported_outputs

    def require_output(self, output_name: str, task_name: str) -> None:
        if self.supports_output(output_name):
            return
        raise ValueError(
            f"Backbone '{self.target}' does not support output '{output_name}' "
            f"required by task='{task_name}'. "
            f"Supported outputs: {list(self.supported_outputs)}"
        )

    def resolve_global_output(self) -> str:
        if self.default_global_output is not None:
            return self.default_global_output
        for output_name in ("cls", "gap", "map"):
            if output_name in self.supported_outputs:
                return output_name
        raise ValueError(
            f"Backbone '{self.target}' does not expose a global output. "
            f"Supported outputs: {list(self.supported_outputs)}"
        )


_BACKBONE_CONTRACTS = {
    "omniprobe.models.clip.CLIP": BackboneContract(
        "omniprobe.models.clip.CLIP",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.convnext.ConvNext": BackboneContract(
        "omniprobe.models.convnext.ConvNext",
        ("dense", "gap"),
        "gap",
        True,
        True,
    ),
    "omniprobe.models.croco.CroCoBackbone": BackboneContract(
        "omniprobe.models.croco.CroCoBackbone",
        ("dense", "gap"),
        "gap",
        True,
        True,
    ),
    "omniprobe.models.c_radio.CRADIOv3Backbone": BackboneContract(
        "omniprobe.models.c_radio.CRADIOv3Backbone",
        ("dense", "gap"),
        "gap",
        True,
        False,
    ),
    "omniprobe.models.c_radio.CRADIOv4Backbone": BackboneContract(
        "omniprobe.models.c_radio.CRADIOv4Backbone",
        ("dense", "gap"),
        "gap",
        True,
        False,
    ),
    "omniprobe.models.deit.DeIT": BackboneContract(
        "omniprobe.models.deit.DeIT",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.stablediffusion.DIFT": BackboneContract(
        "omniprobe.models.stablediffusion.DIFT",
        ("dense", "gap"),
        "gap",
        True,
        True,
    ),
    "omniprobe.models.dino.DINO": BackboneContract(
        "omniprobe.models.dino.DINO",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.dino_diy_sc.DINO": BackboneContract(
        "omniprobe.models.dino_diy_sc.DINO",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.dino_reg.DINOreg": BackboneContract(
        "omniprobe.models.dino_reg.DINOreg",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.dinov3.DinoV3": BackboneContract(
        "omniprobe.models.dinov3.DinoV3",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.dune.DUNE": BackboneContract(
        "omniprobe.models.dune.DUNE",
        ("dense", "gap", "cls"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.ibot.iBOT": BackboneContract(
        "omniprobe.models.ibot.iBOT",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.ijepa.IJEPA": BackboneContract(
        "omniprobe.models.ijepa.IJEPA",
        ("dense", "gap"),
        "gap",
        True,
        True,
    ),
    "omniprobe.models.mae.MAE": BackboneContract(
        "omniprobe.models.mae.MAE",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.metaclip.MetaCLIP": BackboneContract(
        "omniprobe.models.metaclip.MetaCLIP",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.midas_final.make_beit_backbone": BackboneContract(
        "omniprobe.models.midas_final.make_beit_backbone",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.midas_final.MIDAS": BackboneContract(
        "omniprobe.models.midas_final.MIDAS",
        ("dense", "cls", "gap"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.perception.PerceptionBackbone": BackboneContract(
        "omniprobe.models.perception.PerceptionBackbone",
        ("dense", "gap"),
        "gap",
        True,
        True,
    ),
    "omniprobe.models.pixio.PIXIO": BackboneContract(
        "omniprobe.models.pixio.PIXIO",
        ("dense", "gap", "cls"),
        "cls",
        True,
        True,
    ),
    "omniprobe.models.lvlm_visual.QwenVLVisualBackbone": BackboneContract(
        "omniprobe.models.lvlm_visual.QwenVLVisualBackbone",
        ("dense", "gap"),
        "gap",
        False,
        True,
    ),
    "omniprobe.models.lvlm_visual.InternVLVisualBackbone": BackboneContract(
        "omniprobe.models.lvlm_visual.InternVLVisualBackbone",
        ("dense", "gap"),
        "gap",
        False,
        True,
    ),
    "omniprobe.models.lvlm_visual.LlavaOneVisionVisualBackbone": BackboneContract(
        "omniprobe.models.lvlm_visual.LlavaOneVisionVisualBackbone",
        ("dense", "gap"),
        "gap",
        False,
        True,
    ),
    "omniprobe.models.sam.SAM": BackboneContract(
        "omniprobe.models.sam.SAM",
        ("dense",),
        None,
        True,
        True,
    ),
    "omniprobe.models.siglip.SigLIP": BackboneContract(
        "omniprobe.models.siglip.SigLIP",
        ("dense", "gap", "map"),
        "gap",
        True,
        True,
    ),
    "omniprobe.models.vggt.VGGTBackbone": BackboneContract(
        "omniprobe.models.vggt.VGGTBackbone",
        ("dense",),
        None,
        True,
        True,
    ),
    "omniprobe.models.vjepa2.VJEPA2Backbone": BackboneContract(
        "omniprobe.models.vjepa2.VJEPA2Backbone",
        ("dense", "gap"),
        "gap",
        True,
        True,
    ),
}

_CONTRACT_ALIASES = {
    "omniprobe.models.deit.DEIT": "omniprobe.models.deit.DeIT",
    "omniprobe.models.dift_sd.DIFT": "omniprobe.models.stablediffusion.DIFT",
    "omniprobe.models.dinov3.DINOv3": "omniprobe.models.dinov3.DinoV3",
    "omniprobe.models.dune.DuneBackbone": "omniprobe.models.dune.DUNE",
    "omniprobe.models.vggt.VGGT": "omniprobe.models.vggt.VGGTBackbone",
    "omniprobe.models.vjepa2.VJEPA2": "omniprobe.models.vjepa2.VJEPA2Backbone",
}

_MULTILAYER_GLOBAL_TARGETS = {
    "omniprobe.models.stablediffusion.DIFT",
}


def _target_from_cfg(backbone_cfg) -> str:
    if "_target_" not in backbone_cfg:
        raise ValueError("Backbone config is missing '_target_'.")
    return str(backbone_cfg._target_)


def normalize_backbone_target(target: str) -> str:
    if target in _CONTRACT_ALIASES:
        return _CONTRACT_ALIASES[target]
    return target


def uses_multilayer_global_features(backbone_cfg) -> bool:
    return normalize_backbone_target(_target_from_cfg(backbone_cfg)) in _MULTILAYER_GLOBAL_TARGETS


def get_backbone_contract(backbone_cfg, model=None) -> BackboneContract:
    target = normalize_backbone_target(_target_from_cfg(backbone_cfg))
    if "supported_outputs" in backbone_cfg:
        default_global_output = None
        if "default_global_output" in backbone_cfg:
            default_global_output = str(backbone_cfg.default_global_output)
        supports_multilayer = True
        if "supports_multilayer" in backbone_cfg:
            supports_multilayer = bool(backbone_cfg.supports_multilayer)
        supports_layer_selection = True
        if "supports_layer_selection" in backbone_cfg:
            supports_layer_selection = bool(backbone_cfg.supports_layer_selection)
        return BackboneContract(
            target,
            tuple(backbone_cfg.supported_outputs),
            default_global_output,
            supports_multilayer,
            supports_layer_selection,
        )

    if model is not None and hasattr(model, "supported_outputs"):
        supported_outputs = tuple(model.supported_outputs)
        default_global_output = None
        if hasattr(model, "default_global_output"):
            default_global_output = model.default_global_output
        supports_multilayer = True
        if hasattr(model, "supports_multilayer"):
            supports_multilayer = bool(model.supports_multilayer)
        supports_layer_selection = True
        if hasattr(model, "supports_layer_selection"):
            supports_layer_selection = bool(model.supports_layer_selection)
        return BackboneContract(
            target,
            supported_outputs,
            default_global_output,
            supports_multilayer,
            supports_layer_selection,
        )

    if target in _BACKBONE_CONTRACTS:
        return _BACKBONE_CONTRACTS[target]

    if model is None:
        raise ValueError(
            f"No backbone contract registered for '{target}'. Add it to "
            "omniprobe.models.contracts._BACKBONE_CONTRACTS."
        )

    supported_outputs = (str(model.output),)
    supports_multilayer = hasattr(model, "multilayers")
    return BackboneContract(
        target,
        supported_outputs,
        None,
        supports_multilayer,
        True,
    )


def instantiate_backbone_for_output(
    backbone_cfg,
    output_name: str,
    return_multilayer: bool,
    device,
    layer: int = -1,
):
    contract = get_backbone_contract(backbone_cfg)
    contract.require_output(output_name, "runtime")
    if return_multilayer and not contract.supports_multilayer:
        raise ValueError(
            f"Backbone '{contract.target}' does not support multilayer outputs."
        )
    instantiate_cfg = prepare_backbone_instantiate_cfg(backbone_cfg)
    target = get_object(str(instantiate_cfg._target_))
    target_signature = inspect.signature(target)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in target_signature.parameters.values()
    )
    instantiate_kwargs = {
        "output": output_name,
        "return_multilayer": return_multilayer,
    }
    if contract.supports_layer_selection:
        instantiate_kwargs["layer"] = layer
    if not accepts_kwargs:
        instantiate_kwargs = {
            key: value
            for key, value in instantiate_kwargs.items()
            if key in target_signature.parameters
        }
    model = instantiate(
        instantiate_cfg,
        **instantiate_kwargs,
    )
    model = model.to(device)
    model.eval()
    return model, get_backbone_contract(backbone_cfg, model=model)


def prepare_backbone_instantiate_cfg(backbone_cfg):
    instantiate_cfg = OmegaConf.create(OmegaConf.to_container(backbone_cfg, resolve=True))
    for key in (
        "image_mean",
        "supported_outputs",
        "default_global_output",
        "supports_multilayer",
        "supports_layer_selection",
    ):
        if key in instantiate_cfg:
            del instantiate_cfg[key]
    return instantiate_cfg


def resolve_pretrained_backbone_targets() -> set[str]:
    targets = set()
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "backbone"
    for yaml_path in sorted(config_dir.glob("*.yaml")):
        text = yaml_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("_target_:"):
                targets.add(line.split(":", maxsplit=1)[1].strip())
                break
    return targets


def validate_multilayer_feat_dim(model) -> None:
    feat_dim = model.feat_dim
    if isinstance(feat_dim, (int, list, tuple)):
        return
    raise TypeError(
        f"Backbone '{model.__class__.__name__}' has unsupported feat_dim type "
        f"{type(feat_dim)}"
    )

