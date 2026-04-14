from __future__ import annotations

import argparse
import base64
import csv
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from PIL import Image
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)


console = Console()

DEFAULT_OUTPUT_DIR = "/tmp"
DEFAULT_LIMIT = 20
DEFAULT_IMAGE_COLUMN = "preview_path"
DEFAULT_DEVICE = "auto"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_KEEP_ALIVE = "15m"
DEFAULT_OLLAMA_THINK = "false"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
DEFAULT_OLLAMA_TEMPERATURE = 0.0

OLLAMA_SCENE_CAPTION_PROMPT = (
    "Describe this stage-event photo in one short sentence. "
    "Mention the visible people, clothing or dominant colors, and whether it looks most like dance, ceremony, audience, rehearsal, or other. "
    "Do not guess event names, organizations, cities, or venues."
)


@dataclass(frozen=True)
class ImageEntry:
    image_path: Path
    output_name: str
    source_id: str


@dataclass(frozen=True)
class CaptionResult:
    text: str
    metadata: Optional[Mapping[str, Any]] = None


def build_progress_columns() -> tuple[object, ...]:
    return (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    )


def positive_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int_arg(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def resolve_path(base_dir: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def normalize_workspace_relative_path(relative_path: str, column_name: str) -> Path:
    candidate = PurePosixPath(relative_path.strip())
    if not candidate.parts:
        raise ValueError(f"{column_name} is empty")
    if candidate.is_absolute():
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    normalized_parts: List[str] = []
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not normalized_parts:
                raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
            normalized_parts.pop()
            continue
        normalized_parts.append(part)
    if not normalized_parts:
        raise ValueError(f"{column_name} must stay under workspace: {relative_path}")
    return Path(*normalized_parts)


def resolve_workspace_path(workspace_dir: Path, relative_value: str, column_name: str) -> Path:
    workspace_dir = workspace_dir.resolve()
    candidate = normalize_workspace_relative_path(relative_value, column_name)
    resolved_path = (workspace_dir / candidate).resolve()
    try:
        resolved_path.relative_to(workspace_dir)
    except ValueError as exc:
        raise ValueError(f"{column_name} must stay under workspace: {relative_value}") from exc
    return resolved_path


def read_embedded_manifest_entries(
    *,
    index_path: Path,
    workspace_dir: Path,
    image_column: str,
) -> List[ImageEntry]:
    with index_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or ())
        required = {"relative_path", image_column}
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(f"{index_path.name} missing required columns: {', '.join(missing)}")
        entries: List[ImageEntry] = []
        for row in reader:
            image_relative = str(row.get(image_column, "") or "").strip()
            relative_path = str(row.get("relative_path", "") or "").strip()
            if not image_relative or not relative_path:
                continue
            image_path = resolve_workspace_path(workspace_dir, image_relative, image_column)
            entries.append(
                ImageEntry(
                    image_path=image_path,
                    output_name=f"{image_path.name}.txt",
                    source_id=relative_path,
                )
            )
    if not entries:
        raise ValueError(f"{index_path.name} contains no usable rows")
    return entries


def read_gui_index_entries(*, index_path: Path, workspace_dir: Path) -> List[ImageEntry]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    performances = payload.get("performances")
    if not isinstance(performances, list):
        raise ValueError(f"{index_path.name} does not look like a GUI photo index")
    entries: List[ImageEntry] = []
    for performance in performances:
        if not isinstance(performance, Mapping):
            continue
        photos = performance.get("photos")
        if not isinstance(photos, list):
            continue
        for photo in photos:
            if not isinstance(photo, Mapping):
                continue
            proxy_path = str(photo.get("proxy_path", "") or "").strip()
            source_id = str(photo.get("relative_path", "") or "").strip()
            if not proxy_path or not source_id:
                continue
            image_path = resolve_workspace_path(workspace_dir, proxy_path, "proxy_path")
            entries.append(
                ImageEntry(
                    image_path=image_path,
                    output_name=f"{image_path.name}.txt",
                    source_id=source_id,
                )
            )
    if not entries:
        raise ValueError(f"{index_path.name} contains no usable photo entries")
    return entries


def load_image_entries(
    *,
    index_path: Path,
    workspace_dir: Path,
    image_column: str,
    limit: int,
    start_offset: int,
) -> List[ImageEntry]:
    suffix = index_path.suffix.lower()
    if suffix == ".csv":
        entries = read_embedded_manifest_entries(
            index_path=index_path,
            workspace_dir=workspace_dir,
            image_column=image_column,
        )
    elif suffix == ".json":
        entries = read_gui_index_entries(index_path=index_path, workspace_dir=workspace_dir)
    else:
        raise ValueError(f"Unsupported index format: {index_path}")
    sliced = entries[start_offset : start_offset + limit]
    if not sliced:
        raise ValueError("No image entries selected by the requested limit/start-offset")
    return sliced


def write_caption_output(output_dir: Path, output_name: str, text: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    output_path.write_text(text.strip() + "\n", encoding="utf-8")
    return output_path


def extract_ollama_metrics(response_payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for key in (
        "model",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        value = response_payload.get(key)
        if value is not None:
            metrics[key] = value
    return metrics


def render_metric_summary(metrics_list: Sequence[Mapping[str, Any]]) -> str:
    total_count = len(metrics_list)
    total_duration = sum(int(item.get("total_duration", 0) or 0) for item in metrics_list)
    load_duration = sum(int(item.get("load_duration", 0) or 0) for item in metrics_list)
    prompt_eval_count = sum(int(item.get("prompt_eval_count", 0) or 0) for item in metrics_list)
    prompt_eval_duration = sum(int(item.get("prompt_eval_duration", 0) or 0) for item in metrics_list)
    eval_count = sum(int(item.get("eval_count", 0) or 0) for item in metrics_list)
    eval_duration = sum(int(item.get("eval_duration", 0) or 0) for item in metrics_list)
    model_names = sorted({str(item.get("model", "") or "") for item in metrics_list if item.get("model")})
    model_label = ",".join(model_names) if model_names else "unknown"
    avg_total_duration = total_duration / total_count if total_count else 0
    return (
        f"Ollama metrics | model={model_label} samples={total_count} "
        f"total_duration={total_duration} avg_total_duration={avg_total_duration:.1f} "
        f"load_duration={load_duration} prompt_eval_count={prompt_eval_count} "
        f"prompt_eval_duration={prompt_eval_duration} eval_count={eval_count} "
        f"eval_duration={eval_duration}"
    )


def run_caption_cli(
    *,
    day_dir: Path,
    workspace_dir: Path,
    index_path: Path,
    image_column: str,
    limit: int,
    start_offset: int,
    output_dir: Path,
    description: str,
    build_captioner: Callable[[], Callable[[Path], str]],
) -> int:
    entries = load_image_entries(
        index_path=index_path,
        workspace_dir=workspace_dir,
        image_column=image_column,
        limit=limit,
        start_offset=start_offset,
    )
    caption_one = build_captioner()
    metrics_list: List[Mapping[str, Any]] = []
    with Progress(*build_progress_columns(), console=console, expand=False) as progress:
        task = progress.add_task(description.ljust(25), total=len(entries))
        for entry in entries:
            result = caption_one(entry.image_path)
            if isinstance(result, CaptionResult):
                text = result.text
                if result.metadata:
                    metrics_list.append(result.metadata)
            else:
                text = str(result)
            write_caption_output(output_dir, entry.output_name, text)
            progress.advance(task)
    console.print(f"Wrote {len(entries)} caption text file(s) to {output_dir}")
    if metrics_list:
        console.print(render_metric_summary(metrics_list))
    return 0


def choose_torch_device(device: str) -> str:
    if device != DEFAULT_DEVICE:
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def strip_v1_suffix(base_url: str) -> str:
    return base_url[:-3] if base_url.endswith("/v1") else base_url.rstrip("/")


def ollama_post_json(base_url: str, path: str, payload: Dict[str, Any], timeout_seconds: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        strip_v1_suffix(base_url) + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def build_ollama_extra_body(
    *,
    ollama_think: str,
    ollama_num_ctx: Optional[int],
    ollama_num_predict: Optional[int],
) -> Dict[str, Any]:
    extra_body: Dict[str, Any] = {}
    if ollama_think != "inherit":
        reasoning_effort = "none" if ollama_think == "false" else ollama_think
        extra_body["reasoning_effort"] = reasoning_effort
        extra_body["reasoning"] = {"effort": reasoning_effort}
        extra_body["think"] = False if ollama_think == "false" else True
    options: Dict[str, Any] = {}
    if ollama_num_predict is not None:
        options["num_predict"] = ollama_num_predict
    if ollama_num_ctx is not None:
        options["num_ctx"] = ollama_num_ctx
    if options:
        extra_body["options"] = options
    return extra_body


def build_blip_captioner(model_name: str, device: str, max_new_tokens: int) -> Callable[[Path], str]:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    torch_device = choose_torch_device(device)
    processor = BlipProcessor.from_pretrained(model_name, local_files_only=True)
    model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
    model.to(torch_device)
    model.eval()

    def caption_one(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.decode(generated[0], skip_special_tokens=True).strip()

    return caption_one


def build_git_captioner(model_name: str, device: str, max_new_tokens: int) -> Callable[[Path], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch_device = choose_torch_device(device)
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    model.to(torch_device)
    model.eval()

    def caption_one(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return caption_one


def build_florence_captioner(model_name: str, device: str, max_new_tokens: int) -> Callable[[Path], str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch_device = choose_torch_device(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.to(torch_device)
    model.eval()

    def caption_one(image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text="<DETAILED_CAPTION>", images=image, return_tensors="pt")
        inputs = {key: value.to(torch_device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        decoded = processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            decoded,
            task="<DETAILED_CAPTION>",
            image_size=image.size,
        )
        if isinstance(parsed, Mapping):
            value = parsed.get("<DETAILED_CAPTION>") or parsed.get("caption") or ""
            return str(value).strip()
        return str(parsed).strip()

    return caption_one


def build_ollama_captioner(
    *,
    model_name: str,
    ollama_base_url: str,
    ollama_keep_alive: str,
    ollama_think: str,
    ollama_num_ctx: Optional[int],
    ollama_num_predict: Optional[int],
    temperature: float,
    timeout_seconds: float,
) -> Callable[[Path], CaptionResult]:
    extra_body = build_ollama_extra_body(
        ollama_think=ollama_think,
        ollama_num_ctx=ollama_num_ctx,
        ollama_num_predict=ollama_num_predict,
    )

    def caption_one(image_path: Path) -> CaptionResult:
        image_bytes = image_path.read_bytes()
        payload: Dict[str, Any] = {
            "model": model_name,
            "stream": False,
            "keep_alive": ollama_keep_alive,
            "messages": [
                {
                    "role": "user",
                    "content": OLLAMA_SCENE_CAPTION_PROMPT,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "options": {"temperature": temperature},
        }
        if extra_body:
            merged_options = dict(payload["options"])
            merged_options.update(extra_body.get("options", {}))
            payload.update({key: value for key, value in extra_body.items() if key != "options"})
            payload["options"] = merged_options
        response_payload = ollama_post_json(ollama_base_url, "/api/chat", payload, timeout_seconds)
        message = response_payload.get("message")
        if not isinstance(message, Mapping):
            raise ValueError("Ollama response missing message")
        content = str(message.get("content", "") or "").strip()
        if not content:
            raise ValueError("Ollama response returned empty content")
        return CaptionResult(text=content, metadata=extract_ollama_metrics(response_payload))

    return caption_one
