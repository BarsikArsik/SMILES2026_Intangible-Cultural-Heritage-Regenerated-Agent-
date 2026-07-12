# Intangible Cultural Heritage Regenerated Agent

AI-driven pipeline for historically accurate visual reconstruction of intangible cultural heritage artifacts and production processes.

## Overview

This project implements a two-layer agent architecture for generating historically grounded images from heritage descriptions and cultural context.
The workflow follows a reflection pattern: input description and context go to **Layer 1**, prompts are sent to the image model, and **Layer 2** critiques the result and produces a refined prompt when errors are detected.

## Pipeline

```text
Input data (heritage description + context)
    -> Layer 1: Prompt Generator
    -> Flux image model
    -> Layer 2: Critic Agent
       -> if errors detected: refined prompt generation -> Flux
       -> if no errors: final historically accurate image
```

The project presentation describes Layer 1 as the prompt generation stage and Layer 2 as the historical accuracy validation stage.

## Repository structure

```text
.
├── README.md
├── data/
│   ├── examples/
│   ├── contexts/
│   └── taxonomies/
├── prompts/
│   ├── system/
│   └── templates/
├── src/
│   ├── layer1_decomposition/
│   │   ├── __init__.py
│   │   ├── schemas.py
│   │   ├── parser.py
│   │   ├── prompt_builder.py
│   │   └── pipeline.py
│   ├── layer2_critique/
│   │   ├── __init__.py
│   │   ├── validator.py
│   │   ├── critic_prompt.py
│   │   └── pipeline.py
│   ├── flux/
│   │   ├── __init__.py
│   │   └── client.py
│   └── common/
│       ├── __init__.py
│       ├── config.py
│       └── models.py
├── notebooks/
├── tests/
│   ├── test_layer1.py
│   └── test_layer2.py
└── examples/
    ├── demo_layer1.py
    └── demo_layer2.py
```

## Where Layer 1 code should live

Layer 1 is the **decomposition and prompt generation** module, so its code should be placed in `src/layer1_decomposition/`.
According to the presentation, this layer receives heritage description plus context, uses an API call or a local large language model, separates information into four product classes, and produces three main prompts plus five sequential prompts for the manufacturing process.

Suggested files:

- `schemas.py` — Pydantic/dataclass schemas for structured decomposition.
- `parser.py` — converts raw text into normalized structured fields.
- `prompt_builder.py` — functions such as close-up, full-object, and process-step prompt builders.
- `pipeline.py` — orchestrates transcript/context input into prompt packages for Flux.

### Layer 1 responsibilities

- Parse heritage description and cultural context into structured fields.
- Extract at least four categories: object appearance, materials, tools, and manufacturing steps.
- Generate three main prompts and five stepwise prompts for production-stage visualization.
- Return prompt objects in a format ready for Flux input.

### Example Layer 1 API

```python
from src.layer1_decomposition.pipeline import decompose_heritage_description

result = decompose_heritage_description(
    description="Traditional object description...",
    context="Culture, region, historical period..."
)

print(result.main_prompts)
print(result.sequential_prompts)
```

## Where Layer 2 code should live

Layer 2 is the **critique and validation** module, so its code should be placed in `src/layer2_critique/`.
The presentation describes this layer as a Heritage Validator Agent using LangChain and Llama-3.1 via Groq API, with a task of checking whether tools and materials existed in the target era and were characteristic of the specified culture.

Suggested files:

- `validator.py` — validation chain and model wrapper.
- `critic_prompt.py` — system prompts and output format rules for the critic agent.
- `pipeline.py` — receives an initial prompt plus cultural context and returns analysis plus refined prompt.

### Layer 2 responsibilities

- Inspect generated or pre-generation prompts for historical inaccuracies.
- Detect anachronisms in materials, tools, clothing, or process descriptions.
- Preserve visual richness while rewriting the prompt for historical accuracy.
- Return both a short analysis and a refined prompt in English.

### Example Layer 2 API

```python
from src.layer2_critique.pipeline import validate_prompt

review = validate_prompt(
    initial_prompt="A traditional craftsman using a modern metal airbrush...",
    cultural_context="Russian Khokhloma, 17th century..."
)

print(review.analysis)
print(review.refined_prompt)
```

## Suggested data flow between layers

1. Input text and cultural context enter Layer 1.
2. Layer 1 produces structured prompt sets for Flux.
3. Flux generates candidate images or prompt-driven image outputs.
4. Layer 2 checks historical realism and identifies errors.
5. If errors are found, Layer 2 returns a refined prompt and the image is regenerated.
6. If no errors are found, the result is accepted as the final historically accurate image.

## Minimal implementation plan

### `src/layer1_decomposition/`

```python
# pipeline.py
from .parser import parse_description
from .prompt_builder import build_prompt_pack

def decompose_heritage_description(description: str, context: str):
    parsed = parse_description(description, context)
    return build_prompt_pack(parsed)
```

### `src/layer2_critique/`

```python
# pipeline.py
from .validator import run_validator

def validate_prompt(initial_prompt: str, cultural_context: str):
    return run_validator(initial_prompt, cultural_context)
```

## Notes

The attached project slides explicitly identify Layer 1 as **Prompt Generator** and Layer 2 as **Critic Agent**, which supports splitting the repository into two top-level source packages with those responsibilities.
The example for Layer 1 includes prompt-building code around a decomposition schema, while the example for Layer 2 shows a validation function that rewrites historically inaccurate prompts.

## Next steps

- Add transcript ingestion before Layer 1.
- Connect refined prompts directly to the Flux client.
- Build a cultural context database and tag dictionaries for century, region, and tradition metadata.
