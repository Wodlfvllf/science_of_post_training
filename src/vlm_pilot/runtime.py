"""Pinned InternVL/Qwen3 inference with bounded, uncached intervention forwards.

The random offline fixture exercises native Qwen3 attention, not the science.
Real checkpoints execute their pinned native vision/projector and conversation
code. No quantization, batching, Flash conversion or silent device fallback.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .attention import Capture, RouteAction, RoutingSession, Site
from .io import atomic_bytes, canonical, digest, identity
from .schemas import Checkpoint, Inference, PilotError


@dataclass(frozen=True)
class Prepared:
    case: dict
    prompt: str
    input_ids: tuple[int, ...]
    embeddings: torch.Tensor
    image_columns: tuple[int, ...]
    row: int
    pixels: torch.Tensor | None = None


class FixtureTokenizer:
    """Fixed word vocabulary; never learned from labels or expanded during a run."""

    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    _pattern = re.compile(r"<\|im_start\|>|<\|im_end\|>|<IMG_CONTEXT>|</?img>|"
                          r"[A-Za-z_]+|[0-9]+|\n|[^\w\s]|[^\S\n]+")

    def __init__(self):
        words = ("You are a helpful assistant system user Starting at the follow one arrow "
                 "forward backward What color is destination chip Reply with name shape socket "
                 "Text only control Chips Arrows has to red blue green yellow purple orange "
                 "triangle circle square Answer "
                 "<|im_start|> <|im_end|> <IMG_CONTEXT> <img> </img>").split()
        tokens = sorted(set(words) | {" ", "\n", ".", ":", ";", "-", "?", ","})
        self.vocab = {token: index + 4 for index, token in enumerate(tokens)}
        self.reverse = {value: key for key, value in self.vocab.items()}

    def encode(self, text, add_special_tokens=True):
        tokens = [self.vocab.get(" " if token.isspace() and token != "\n" else token, 3)
                  for token in self._pattern.findall(text)]
        return [self.bos_token_id, *tokens] if add_special_tokens else tokens

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.reverse.get(int(i), "<unk>") for i in ids
                       if not skip_special_tokens or int(i) not in (0, 1, 2))

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 3)

    def get_vocab(self):
        return dict(self.vocab)


class FixtureVisionModel(nn.Module):
    """Random patch-color projector, not a trained vision encoder."""

    def __init__(self, hidden_size):
        super().__init__()
        self.projector = nn.Linear(3, hidden_size)

    def forward(self, pixels):
        patches = torch.nn.functional.adaptive_avg_pool2d(pixels, (16, 16))
        return self.projector(patches.flatten(2).transpose(1, 2))


class FixtureModel(nn.Module):
    def __init__(self, checkpoint):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        super().__init__()
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            config = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                                 num_hidden_layers=2, num_attention_heads=4,
                                 num_key_value_heads=2, head_dim=8,
                                 max_position_embeddings=4096, attention_dropout=0.,
                                 use_cache=False, attn_implementation="eager")
            self.language_model = Qwen3ForCausalLM(config)
            self.vision_model = FixtureVisionModel(32)
            if checkpoint.stage == "gspo":
                generator = torch.Generator().manual_seed(checkpoint.fixture_seed)
                with torch.no_grad():
                    for layer in self.language_model.model.layers:
                        for name in ("q_proj", "k_proj"):
                            weight = getattr(layer.self_attn, name).weight
                            weight.add_(torch.randn(weight.shape, generator=generator) * .01)
        self.num_image_token, self.ps_version = 256, "v2"
        self.template, self.system_message = "internvl2_5", "You are a helpful assistant."
        self.img_context_token_id = None

    def extract_feature(self, pixels):
        return self.vision_model(pixels)

    def forward(self, prepared):
        ids = torch.tensor([prepared.input_ids], device=prepared.embeddings.device)
        embeddings = self.language_model.get_input_embeddings()(ids).clone()
        if prepared.pixels is not None:
            embeddings[:, list(prepared.image_columns)] = self.extract_feature(prepared.pixels)
        return self.language_model(inputs_embeds=embeddings, use_cache=False,
                                   output_attentions=False, output_hidden_states=False).logits[:, -1:]


def runtime_profile(device) -> dict:
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "Pillow",
                 "pydantic", "PyYAML", "filelock"):
        packages[name] = version(name)
    result = {"python": sys.version.split()[0], "system": platform.system(),
              "machine": platform.machine(), "packages": packages, "device": str(device),
              "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
              "matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic": torch.are_deterministic_algorithms_enabled(),
              "default_dtype": str(torch.get_default_dtype()), "cuda": torch.version.cuda}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu=properties.name, total_memory=properties.total_memory,
                      capability=list(torch.cuda.get_device_capability(device)),
                      tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                      tf32_cudnn=torch.backends.cudnn.allow_tf32,
                      cudnn_benchmark=torch.backends.cudnn.benchmark,
                      cudnn_deterministic=torch.backends.cudnn.deterministic)
    return result


def gpu_identifier(device) -> str:
    """Use the logical CUDA device's UUID, not an assumed nvidia-smi ordinal."""
    properties = torch.cuda.get_device_properties(device)
    value = getattr(properties, "uuid", None)
    if isinstance(value, bytes):
        value = value.decode("ascii")
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9-]+", value):
        return value if value.startswith(("GPU-", "MIG-")) else "GPU-" + value
    raise PilotError("Cannot identify the selected logical GPU UUID; verify CUDA visibility.")


def _weight_files(checkpoint, allow_downloads) -> list[dict]:
    from .weights import SafetensorSource

    source = SafetensorSource(checkpoint.model_id, checkpoint.revision, allow_downloads)
    result = []
    for name in sorted(set(source._load_index().values())):
        path = source._download(name)
        size = source._file_size(path, source.MAX_SHARD_BYTES, "loaded weight shard")
        checksum = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024**2):
                checksum.update(chunk)
        result.append({"name": name, "size_bytes": size, "sha256": checksum.hexdigest()})
    return result


def _tokenizer_assets(checkpoint, allow_downloads) -> dict:
    """Bind merges, added tokens and processor/template assets, not vocabulary alone."""
    from huggingface_hub import hf_hub_download

    from .weights import SafetensorSource

    source = SafetensorSource(checkpoint.model_id, checkpoint.revision, allow_downloads)
    index_file = Path(hf_hub_download(checkpoint.model_id, "config.json", revision=checkpoint.revision,
                                    local_files_only=not allow_downloads))
    result = {"config.json": digest(index_file.read_bytes())}
    for name in ("tokenizer_config.json", "vocab.json", "merges.txt", "added_tokens.json",
                 "special_tokens_map.json", "conversation.py", "configuration_internvl_chat.py",
                 "configuration_intern_vit.py"):
        file = source._download(name)
        if file.stat().st_size > 32 * 1024**2:
            raise PilotError("Pinned tokenizer/model-code asset exceeds its bounded size.")
        result[name] = digest(file.read_bytes())
    return result


class Engine:
    def __init__(self, checkpoint: Checkpoint, inference: Inference, *,
                 allow_downloads=False, device: str | None = None):
        if version("transformers") != "4.57.6":
            raise PilotError("The intervention adapter requires Transformers 4.57.6; use vlm extra.")
        self.checkpoint, self.settings = checkpoint, inference
        if re.fullmatch(r"cpu|cuda:[0-9]+", device or inference.device) is None:
            raise PilotError("Select cpu or cuda:N explicitly; no MPS/auto/offload fallback.")
        self.device = torch.device(device or inference.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise PilotError("CUDA was explicitly selected but is unavailable; no CPU fallback.")
        if self.device.type == "cuda":
            executable = shutil.which("nvidia-smi")
            if executable is None:
                raise PilotError("Cannot check GPU occupancy: nvidia-smi is missing.")
            processes = subprocess.run([executable, "-i", gpu_identifier(self.device),
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15, check=False)
            if processes.returncode != 0:
                raise PilotError("Cannot verify selected GPU occupancy; inspect it before running.")
            pids = [line.strip() for line in processes.stdout.splitlines() if line.strip()]
            if any(pid != str(os.getpid()) for pid in pids):
                raise PilotError("Selected GPU has another compute process. Choose a free cuda:N; "
                                 "the active maths/repair worker will not be interrupted.")
        if checkpoint.backend == "fixture" and self.device.type != "cpu":
            raise PilotError("The meaningless random fixture is CPU-only.")
        self.dtype = getattr(torch, inference.dtype)
        start = time.perf_counter()
        if checkpoint.backend == "fixture":
            self.model, self.tokenizer = FixtureModel(checkpoint), FixtureTokenizer()
        else:
            from transformers import AutoModel, AutoTokenizer

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint.model_id, revision=checkpoint.revision,
                    trust_remote_code=True, use_fast=False, local_files_only=not allow_downloads,
                )
                self.model = AutoModel.from_pretrained(
                    checkpoint.model_id, revision=checkpoint.revision,
                    code_revision=checkpoint.revision, trust_remote_code=True,
                    torch_dtype=self.dtype, low_cpu_mem_usage=True,
                    device_map={"": str(self.device)},
                    use_safetensors=True,
                    use_flash_attn=False, local_files_only=not allow_downloads,
                )
            except Exception as error:
                raise PilotError(f"Pinned InternVL load failed ({type(error).__name__}); "
                                 "check the vlm environment/cache or enable downloads.") from None
        self.model = self.model.to(device=self.device, dtype=self.dtype).eval()
        self.language_model = self.model.language_model
        self.language_model.config.use_cache = False
        self.language_model.config.output_attentions = False
        self.language_model.config.output_hidden_states = False
        if (self.language_model.config._attn_implementation != "eager"
                or type(self.language_model).__name__ != "Qwen3ForCausalLM"
                or self.model.num_image_token != 256 or self.model.ps_version != "v2"):
            raise PilotError("Only dense native InternVL3.5/Qwen3, eager, 256-token v2 tiles work.")
        self.img_token = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if self.img_token is None or self.img_token == getattr(self.tokenizer, "unk_token_id", None):
            raise PilotError("Pinned tokenizer has no actual image-context token.")
        self.model.img_context_token_id = self.img_token
        config = self.language_model.config.to_dict()
        code = {}
        for obj in (type(self.model), type(self.model.vision_model), type(self.language_model)):
            path = Path(inspect.getfile(obj))
            code[f"{obj.__module__}.{obj.__name__}"] = digest(path.read_bytes())
        self.model_profile = {
            "checkpoint": checkpoint.model_dump(mode="json"), "dtype": str(self.dtype),
            "lm_config_sha256": identity(config), "tokenizer_sha256": identity(self.tokenizer.get_vocab()),
            "model_code_sha256": code, "image_tokens": 256, "tile_grid": [16, 16],
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "scientific_fixture": checkpoint.backend == "fixture",
            "weight_files": _weight_files(checkpoint, allow_downloads)
            if checkpoint.backend == "internvl" else [],
            "tokenizer_assets": _tokenizer_assets(checkpoint, allow_downloads)
            if checkpoint.backend == "internvl" else {},
            "common_inference": inference.model_dump(mode="json"),
            "native_template": self.model.template,
            "native_system_message": self.model.system_message,
        }
        self.runtime_profile = runtime_profile(self.device)
        self.load_seconds = time.perf_counter() - start
        self._family = None
        self._features: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def _prompt(self, question: str, has_image: bool, profile: str) -> str:
        content = ("<img>" + "<IMG_CONTEXT>" * 256 + "</img>\n" if has_image else "") + question
        if self.checkpoint.backend == "fixture":
            prompt = (f"<|im_start|>system\n{self.settings.system_message}<|im_end|>\n"
                      f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
        else:
            module = importlib.import_module(type(self.model).__module__)
            template = module.get_conv_template(self.model.template)
            template.system_message = (self.settings.system_message if profile == "common"
                                       else self.model.system_message)
            template.append_message(template.roles[0], content)
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt()
        return prompt + (self.settings.answer_prefix if profile == "common" else "")

    def prepare(self, root: Path, family: dict, case: dict, *, profile="common") -> Prepared:
        if self._family != family["family_id"]:
            self._features.clear()
            self._family = family["family_id"]
        prompt = self._prompt(case["question"], case["image_id"] is not None, profile)
        ids = tuple(self.tokenizer.encode(prompt, add_special_tokens=True))
        if not ids or len(ids) >= self.settings.max_context_tokens:
            raise PilotError("Prompt exceeds the declared context; no silent truncation.")
        columns = tuple(i for i, token in enumerate(ids) if token == self.img_token)
        if len(columns) != (256 if case["image_id"] is not None else 0):
            raise PilotError("Rendered image-context token count disagrees with the native tile.")
        with torch.no_grad():
            embeddings = self.language_model.get_input_embeddings()(
                torch.tensor([ids], device=self.device)).clone()
            pixels = None
            if case["image_id"] is not None:
                image_id = case["image_id"]
                if image_id not in self._features:
                    from PIL import Image

                    item = next(i for i in family["images"] if i["image_id"] == image_id)
                    path = root / item["relative_path"]
                    if digest(path.read_bytes()) != item["sha256"]:
                        raise PilotError("Image bytes changed after plan locking.")
                    with Image.open(path) as image:
                        image = image.convert("RGB").resize((448, 448), Image.Resampling.BICUBIC)
                        array = np.asarray(image, dtype=np.float32).copy() / 255.
                    pixels = torch.from_numpy(array).permute(2, 0, 1)
                    mean = torch.tensor((.485, .456, .406))[:, None, None]
                    std = torch.tensor((.229, .224, .225))[:, None, None]
                    pixels = ((pixels - mean) / std).unsqueeze(0).to(self.device, self.dtype)
                    features = self.model.extract_feature(pixels)
                    if features.shape != (1, 256, embeddings.shape[-1]) or not torch.isfinite(features).all():
                        raise PilotError("Native vision/projector produced invalid tile features.")
                    self._features[image_id] = (features.detach(), pixels)
                features, pixels = self._features[image_id]
                embeddings[:, list(columns)] = features
        return Prepared(case, prompt, ids, embeddings, columns, len(ids) - 1, pixels)

    @staticmethod
    def require_alignment(recipient: Prepared, donor: Prepared, *, allow_query_change=False):
        if (len(recipient.input_ids) != len(donor.input_ids) or recipient.row != donor.row
                or recipient.image_columns != donor.image_columns):
            raise PilotError("Donor prompt/token/rotary positions do not align. Use a new locked prompt.")
        if not allow_query_change and recipient.input_ids != donor.input_ids:
            raise PilotError("Image-only donor must use exactly the recipient prompt tokens.")

    def _adapt_action(self, action: RouteAction, length: int) -> RouteAction:
        """Append-only candidate prefixes cannot influence the causal pre-answer row."""
        fields = {}
        for name in ("query", "key", "value", "distribution"):
            capture = getattr(action, name)
            if capture is None or capture.sequence_length == length:
                continue
            extra = length - capture.sequence_length
            if extra < 0:
                raise PilotError("Donor cannot be truncated to align a candidate prefix.")
            positions = (torch.arange(length)[None] if capture.position_ids is not None else None)
            mask = torch.cat((capture.mask_row, torch.full(
                (*capture.mask_row.shape[:-1], extra), torch.finfo(capture.mask_row.dtype).min)), -1)
            fields[name] = replace(capture, sequence_length=length, position_ids=positions, mask_row=mask)
        return replace(action, **fields)

    def forward(self, prepared: Prepared, *, sites=(), actions=None, keep=1):
        length = prepared.embeddings.shape[1]
        if length > self.settings.max_context_tokens:
            raise PilotError("Candidate/generation exceeds the declared context.")
        session = None
        if sites:
            if not prepared.image_columns:
                raise PilotError("Routing interventions require actual image columns.")
            converted = {site: self._adapt_action(action, length) for site, action in (actions or {}).items()}
            session = RoutingSession(self.language_model, tuple(sites), row=prepared.row,
                                     image_columns=prepared.image_columns, actions=converted)
        with torch.no_grad():
            kwargs = dict(inputs_embeds=prepared.embeddings,
                          attention_mask=torch.ones((1, length), dtype=torch.long, device=self.device),
                          position_ids=torch.arange(length, device=self.device)[None],
                          use_cache=False, output_attentions=False, output_hidden_states=False,
                          return_dict=True, logits_to_keep=keep)
            if session is None:
                logits = self.language_model(**kwargs).logits
            else:
                with session:
                    logits = self.language_model(**kwargs).logits
        if not torch.isfinite(logits).all():
            raise PilotError("Nonfinite native/intervened logits; no record earns completion.")
        return logits, (session.captures if session else {}), (session.diagnostics if session else {})

    def suffixes(self, prepared: Prepared) -> dict[str, tuple[int, ...]]:
        result = {}
        for candidate in prepared.case["candidates"]:
            suffix = (" " if self.settings.answer_prefix and prepared.prompt.endswith(
                self.settings.answer_prefix) else "") + candidate
            combined = tuple(self.tokenizer.encode(prepared.prompt + suffix, add_special_tokens=True))
            if combined[:len(prepared.input_ids)] != prepared.input_ids:
                raise PilotError("Candidate rewrites conditioning tokens; change and relock the prefix.")
            tokens = combined[len(prepared.input_ids):]
            if not tokens:
                raise PilotError("Empty candidate continuation.")
            result[candidate] = tokens
        return result

    def score(self, prepared: Prepared, *, sites=(), actions=None):
        suffixes = self.suffixes(prepared)
        start = time.perf_counter()
        logits, captures, diagnostics = self.forward(prepared, sites=sites, actions=actions)
        first = torch.log_softmax(logits[0, -1].float(), -1)
        scores, token_scores = {}, {}
        forwards = 1
        for candidate, tokens in suffixes.items():
            values = [float(first[tokens[0]].item())]
            if len(tokens) > 1:
                extension = torch.tensor([tokens[:-1]], device=self.device)
                with torch.no_grad():
                    extended = replace(prepared, embeddings=torch.cat((prepared.embeddings,
                        self.language_model.get_input_embeddings()(extension)), 1))
                extra_logits, _, _ = self.forward(extended, sites=sites, actions=actions,
                                                  keep=len(tokens))
                logprobs = torch.log_softmax(extra_logits[0].float(), -1)
                values = [float(logprobs[i, token].item()) for i, token in enumerate(tokens)]
                forwards += 1
            scores[candidate], token_scores[candidate] = sum(values), values
        maximum = max(scores.values())
        winners = [key for key, value in scores.items() if value == maximum]
        result = {"scores": scores, "candidate_token_ids": {k: list(v) for k, v in suffixes.items()},
                  "candidate_token_logprobs": token_scores,
                  "prediction": winners[0] if len(winners) == 1 else None,
                  "tied": len(winners) != 1, "forward_passes": forwards,
                  "elapsed_seconds": time.perf_counter() - start,
                  "input_ids": list(prepared.input_ids), "rendered_prompt": prepared.prompt,
                  "row": prepared.row, "image_columns": list(prepared.image_columns),
                  "pixel_to_token": {"image_size": 448, "grid": [16, 16], "ps_version": "v2"},
                  "routing": {f"{s.layer}:{s.kv_group}": v for s, v in diagnostics.items()},
                  "score_kind": "complete_candidate_sequence_logprob"}
        return result, captures

    def generate(self, prepared: Prepared, budget: int) -> dict:
        """Unrestricted greedy audit; every step recomputes the full uncached prefix."""
        start = time.perf_counter()
        if len(prepared.input_ids) + budget > self.settings.max_context_tokens:
            raise PilotError("Generation budget exceeds the locked context.")
        current, tokens = prepared, []
        eos = self.tokenizer.eos_token_id
        for _ in range(budget):
            logits, _, _ = self.forward(current)
            token = int(logits[0, -1].argmax().item())
            tokens.append(token)
            if token == eos:
                break
            with torch.no_grad():
                embedding = self.language_model.get_input_embeddings()(
                    torch.tensor([[token]], device=self.device))
            current = replace(current, embeddings=torch.cat((current.embeddings, embedding), 1))
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        parsed = text.strip().lower().rstrip(".").strip()
        valid = parsed in prepared.case["candidates"] and tokens[-1] == eos
        return {"text": text, "token_ids": tokens, "finish_reason": "eos" if tokens[-1] == eos else "length",
                "valid": valid, "parsed_answer": parsed if valid else None,
                "correct": valid and parsed == prepared.case["gold"],
                "elapsed_seconds": time.perf_counter() - start,
                "settings": {"max_new_tokens": budget, "do_sample": False, "use_cache": False,
                             "eos_token_id": eos, "profile": "unrestricted_greedy_audit"}}

    def preflight(self, root: Path, family: dict, case: dict, sites=()) -> dict:
        prepared = self.prepare(root, family, case)
        sites = tuple(sites) or (Site(0, 0),)
        native, _, _ = self.forward(prepared)
        captured, captures, diagnostics = self.forward(prepared, sites=sites)
        self_actions = {site: RouteAction("qk", query=captures[site], key=captures[site]) for site in sites}
        patched, _, _ = self.forward(prepared, sites=sites, actions=self_actions)
        with torch.no_grad():
            if self.checkpoint.backend == "fixture":
                reference = self.model(prepared)
            else:
                ids = torch.tensor([prepared.input_ids], device=self.device)
                reference = self.model(pixel_values=prepared.pixels, input_ids=ids,
                    attention_mask=torch.ones_like(ids), image_flags=torch.ones((1, 1),
                    dtype=torch.long, device=self.device), use_cache=False, output_attentions=False,
                    output_hidden_states=False, return_dict=True).logits[:, -1:]
        candidate_checks = {}
        suffixes = self.suffixes(prepared)
        _, initial_captures, _ = self.forward(prepared, sites=sites)
        for candidate, tokens in suffixes.items():
            if len(tokens) <= 1:
                continue
            extension = torch.tensor([tokens[:-1]], device=self.device)
            with torch.no_grad():
                extended = replace(prepared, embeddings=torch.cat((prepared.embeddings,
                    self.language_model.get_input_embeddings()(extension)), 1))
            plain, _, _ = self.forward(extended, keep=len(tokens))
            capture_only, _, _ = self.forward(extended, sites=sites, keep=len(tokens))
            actions = {site: RouteAction("distribution", distribution=initial_captures[site]) for site in sites}
            self_extended, _, _ = self.forward(extended, sites=sites, actions=actions, keep=len(tokens))
            # Full-sequence and last-logit optimization must have equivalent candidate scores.
            deltas = [(capture_only.float()-plain.float()).abs(),
                      (self_extended.float()-plain.float()).abs()]
            tolerance = self.settings.atol + self.settings.rtol * plain.float().abs()
            candidate_checks[candidate] = {"passed": all(bool((delta <= tolerance).all()) for delta in deltas),
                                           "max_abs": max(float(delta.max()) for delta in deltas)}
        checks = {}
        for label, actual in (("native_embedding", reference), ("capture_only", captured),
                              ("self_qk", patched)):
            finite = bool(torch.isfinite(actual).all())
            delta = (actual.float() - native.float()).abs()
            tolerance = self.settings.atol + self.settings.rtol * native.float().abs()
            failed = int((delta > tolerance).sum().item())
            argmax_match = bool(torch.equal(actual.argmax(-1), native.argmax(-1)))
            checks[label] = {"finite": finite, "mismatches": failed,
                             "max_abs": float(delta.max().item()),
                             "argmax_match": argmax_match,
                             "passed": finite and failed == 0 and argmax_match}
        reconstruction = all(d["reconstruction_weight_maxdiff"] <= self.settings.atol
                             for d in diagnostics.values())
        report = {"passed": all(c["passed"] for c in checks.values()) and reconstruction
                    and all(c["passed"] for c in candidate_checks.values()),
                  "checks": checks, "routing": {f"{s.layer}:{s.kv_group}": d
                                                  for s, d in diagnostics.items()},
              "candidate_suffixes": {k: list(v) for k, v in suffixes.items()},
              "multi_token_candidate_checks": candidate_checks,
                  "load_seconds": self.load_seconds,
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated(self.device)
                  if self.device.type == "cuda" else None,
                  "runtime": self.runtime_profile, "model": self.model_profile,
                  "fixture_only": self.checkpoint.backend == "fixture"}
        return report


def save_capture(root: Path, trial_id: str, captures: dict[Site, Capture]) -> list[dict]:
    from safetensors.torch import save

    refs = []
    for site, capture in captures.items():
        tensors = {name: getattr(capture, name) for name in (
            "q", "k", "v", "conditional", "mass", "cos_row", "sin_row", "cos_image", "sin_image",
            "input_rows", "input_image", "mask_row", "position_ids") if getattr(capture, name) is not None}
        payload = save(tensors, metadata={"capture": canonical(capture.metadata).decode()})
        if len(payload) > 64 * 1024**2:
            raise PilotError("Capture exceeds the 64 MiB bounded artifact size.")
        sha = digest(payload)
        name = f"captures/{trial_id}-{site.layer}-{site.kv_group}-{sha}.safetensors"
        path = root / name
        if not path.exists():
            atomic_bytes(path, payload, immutable=True)
        elif digest(path.read_bytes()) != sha:
            raise PilotError("Existing capture bytes disagree with their digest.")
        refs.append({"path": name, "sha256": sha, "size_bytes": len(payload)})
    return refs