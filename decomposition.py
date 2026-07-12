import os
import json
import argparse
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class DecompositionSchema(BaseModel):
    appearance: str = Field(description="A short but informative visual description of the item")
    tools: List[str] = Field(default_factory=list, description="Manufacturing tools only if explicitly stated in the text")
    materials: List[str] = Field(default_factory=list, description="Subject materials only if explicitly stated in the text")
    decorative_elements: List[str] = Field(default_factory=list, description="Individual decorative elements only if they are clearly indicated in the text")
    manufacturing_steps: List[str] = Field(description="Exactly 5 steps of production")


SYSTEM_INSTRUCTION = """
You are analyzing a long ethnographic, museum, archaeological, or oral description of a cultural heritage item.
You must decompose it and return strictly structured JSON.

Rules:
1. Don't invent facts that aren't in the text.
2. If tools, materials, or decorative elements aren't explicitly stated, return an empty list.
3. The appearance field must describe the visible appearance of the item.
4. manufacturing_steps must contain exactly 5 steps.
5. If the text is incomplete or is recorded speech, carefully normalize the process to 5 steps, without going beyond what was said.
6. Don't return markdown, explanations, or text outside of JSON.
7. Write briefly, precisely, and to the point.
""".strip()


USER_TEMPLATE = """
Decompose the description of a cultural heritage item using the diagram.

Extract:
1. description of the item's appearance,
2. manufacturing tools,
3. materials,
4. individual decorative elements,
5. five manufacturing steps.

Text for analysis:
{input_text}
""".strip()


def analyze_with_gemini(input_text: str, model_name: str = "gemini-2.5-flash") -> DecompositionSchema:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GEMINI_API_KEY environment variable before running the script.")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=USER_TEMPLATE.format(input_text=input_text),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=DecompositionSchema,
            temperature=0.2,
        ),
    )

    if not response.parsed:
        raise ValueError(f"Gemini did not return parsed structured output. Raw response: {response.text}")

    result = response.parsed
    if len(result.manufacturing_steps) != 5:
        raise ValueError(f"Expected exactly 5 manufacturing steps, got {len(result.manufacturing_steps)}")
    return result


def build_main_flux_prompt(parsed: DecompositionSchema) -> str:
    materials_text = ", ".join(parsed.materials) if parsed.materials else "historically plausible traditional materials"
    decor_text = ", ".join(parsed.decorative_elements) if parsed.decorative_elements else "subtle handcrafted decorative details"
    tools_text = ", ".join(parsed.tools) if parsed.tools else "traditional hand tools"
    return (
        f"A museum-quality documentary rendering of a cultural heritage object. "
        f"Appearance: {parsed.appearance}. "
        f"The object is made from {materials_text}, with visible handcrafted construction, realistic seams, natural surface wear, and accurate material texture. "
        f"The manufacturing process involves {tools_text}. "
        f"Decorative elements include {decor_text}, arranged in a historically plausible way. "
        f"Neutral museum background, soft directional lighting, high detail, ethnographic realism, sharp texture definition, natural color balance."
    )


def build_object_sheet_prompt(parsed: DecompositionSchema) -> str:
    materials_text = ", ".join(parsed.materials) if parsed.materials else "traditional materials"
    decor_text = ", ".join(parsed.decorative_elements) if parsed.decorative_elements else "handcrafted ornament"
    return (
        "A clean catalog-style object sheet of a cultural heritage artifact, isolated against a plain neutral background. "
        f"The object has the following appearance: {parsed.appearance}. "
        f"Materials: {materials_text}. Decorative elements: {decor_text}. "
        "Front-facing museum documentation view, even soft lighting, maximum material readability, no dramatic scene, no clutter."
    )


def build_closeup_prompt(parsed: DecompositionSchema) -> str:
    materials_text = ", ".join(parsed.materials) if parsed.materials else "traditional materials"
    decor_text = ", ".join(parsed.decorative_elements) if parsed.decorative_elements else "handcrafted ornament"
    return (
        "A high-detail close-up study of a cultural heritage object surface. "
        f"Focus on materials: {materials_text}. Decorative detail: {decor_text}. "
        "Macro-like documentary view, tactile surface realism, visible stitching, wear marks, pigment traces, shallow depth of field, soft museum lighting."
    )


def build_process_prompts(parsed: DecompositionSchema) -> List[str]:
    materials_text = ", ".join(parsed.materials) if parsed.materials else "traditional raw materials"
    prompts = []
    for i, step in enumerate(parsed.manufacturing_steps, start=1):
        prompts.append(
            f"A documentary illustration of step {i} in the making of a traditional cultural heritage object. "
            f"Object type and appearance: {parsed.appearance}. Materials: {materials_text}. "
            f"This step shows: {step} Neutral workshop or ethnographic reconstruction setting, realistic hand tools, high detail, natural light, material accuracy."
        )
    return prompts


def write_text(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


def save_outputs(parsed: DecompositionSchema, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)

    decomposition_path = output_dir / "decomposition.json"
    decomposition_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")

    write_text(prompts_dir / "01_main_flux_prompt.txt", build_main_flux_prompt(parsed))
    write_text(prompts_dir / "02_object_sheet_prompt.txt", build_object_sheet_prompt(parsed))
    write_text(prompts_dir / "03_closeup_prompt.txt", build_closeup_prompt(parsed))

    for i, prompt in enumerate(build_process_prompts(parsed), start=1):
        write_text(prompts_dir / f"process_step_{i:02d}.txt", prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini-based GenArtist-style decomposition from a .txt file.")
    parser.add_argument("input_txt", help="Path to .txt file with long description")
    parser.add_argument("-o", "--output", default="gemini_genartist_output", help="Output directory")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    args = parser.parse_args()

    input_path = Path(args.input_txt)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    input_text = input_path.read_text(encoding="utf-8").strip()
    if not input_text:
        raise ValueError("Input text file is empty")

    parsed = analyze_with_gemini(input_text, model_name=args.model)
    save_outputs(parsed, Path(args.output))

    print(f"Input: {input_path}")
    print(f"Model: {args.model}")
    print(f"Output folder: {Path(args.output).resolve()}")
    print("Created files:")
    for p in sorted(Path(args.output).rglob("*.txt")):
        print(f" - {p}")
    print(f" - {Path(args.output) / 'decomposition.json'}")


if __name__ == "__main__":
    main()
