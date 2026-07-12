# Intangible Cultural Heritage Regenerated Agent

AI-driven pipeline for historically accurate visual reconstruction of intangible cultural heritage artifacts and production processes.
Developed by:
@arsenii_galimov (Arsenii Galimov)
@evekozlova (Eva Kozlova)

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



## Layer 1 responsibilities
Layer 1 is the **decomposition and prompt generation** module

- Parse heritage description and cultural context into structured fields.
- Extract at least four categories: object appearance, materials, tools, and manufacturing steps.
- Generate three main prompts and five stepwise prompts for production-stage visualization.
- Return prompt objects in a format ready for Flux input.



## Layer 2 responsibilities

Layer 2 is the **critique and validation** module.

- Inspect generated or pre-generation prompts for historical inaccuracies.
- Detect anachronisms in materials, tools, clothing, or process descriptions.
- Preserve visual richness while rewriting the prompt for historical accuracy.
- Return both a short analysis and a refined prompt in English.


## Next steps

- Add transcript ingestion before Layer 1.
- Connect refined prompts directly to the Flux client.
- Build a cultural context database and tag dictionaries for century, region, and tradition metadata.
