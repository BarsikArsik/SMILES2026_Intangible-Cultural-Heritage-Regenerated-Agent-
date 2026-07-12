import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

os.environ["GROQ_API_KEY"] = "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"
llm_groq = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

class DecompositionSchema(BaseModel):
    appearance: str = Field(description="A short but informative visual description of the item")
    tools: List[str] = Field(default_factory=list, description="Manufacturing tools only if explicitly stated in the text")
    materials: List[str] = Field(default_factory=list, description="Subject materials only if explicitly stated in the text")
    decorative_elements: List[str] = Field(default_factory=list, description="Individual decorative elements only if they are clearly indicated in the text")
    manufacturing_steps: List[str] = Field(description="Exactly 5 steps of production")

def analyze_with_groq(input_text: str) -> DecompositionSchema:
    parser = PydanticOutputParser(pydantic_object=DecompositionSchema)
    template = ChatPromptTemplate.from_template(
        "You are analyzing an ethnographic description of a cultural heritage item.\n"
        "Decompose it and return strictly structured JSON.\n"
        "Extract: 1. appearance, 2. manufacturing tools, 3. materials, 4. decorative elements, 5. five manufacturing steps.\n"
        "Text for analysis: {input_text}\n"
        "{format_instructions}"
    )
    chain = template | llm_groq | parser
    return chain.invoke({"input_text": input_text, "format_instructions": parser.get_format_instructions()})

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

input_text = """
Folk decorative art traditionally encompasses regions of the Far East. Women were involved in the artistic 
processing of items from soft materials such as fish skin, fur, hides, fabrics, and also from birch bark. 
Fish skin has been used for centuries. The museum's collection contains: a robe by the Ulchi craftswoman Z. Plastina. 
On the wrap-around kimono-type robe, in the tradition of embellishing the collar, the edge of the left flap, and the hem. 
The base is glued onto them with straight and jagged strips of darker-colored fish skin. The back of the robe is decorated 
with a series of appliqué images. The painted ornaments convey a harmonious system of the universe. 
The human figures likely personified cultural heroes or ancestors, patron spirits. Dragons occupy a central place 
in the pictorial art and mythology of the peoples of the Far East.
"""

context = """
Far East folk decorative art, Ulchi and Nivkh cultures. 
Traditional materials: natural fish skin, sinew (threads made from fish skin), plant-based dyes. 
Traditional tools: hand-stitching, appliqué. 
No synthetic materials (polyester, plastic) or modern machinery (sewing machines) were used.
"""

parsed_data = analyze_with_groq(input_text)
layer1_prompt = build_main_flux_prompt(parsed_data)

print("Layer 1 Prompt:")
print(layer1_prompt)

def validate_prompt(initial_prompt, cultural_context):
    template = ChatPromptTemplate.from_template(
        "You are a strict Level 2 Validation Agent. Your task is to validate realism, compare against ethnographic descriptions, "
        "and evaluate production logic. Analyze the visual prompt for Flux.\n"
        "Prompt: {prompt}\n"
        "Ethnographic Context: {context}\n"
        "Check by criteria: did materials and tools exist in the specified era, are they characteristic of this culture. "
        "If there are anachronisms or modern errors, correct the prompt for historical accuracy while preserving visual expressiveness. "
        "Respond strictly in the following format:\n"
        "Analysis: [Brief error analysis]\n"
        "Refined prompt: [Final corrected prompt in English]\n"
        "Do not write any additional notes, introductions, or conclusions. Only the specified format."
    )
    chain = template | llm_groq | StrOutputParser()
    return chain.invoke({"prompt": initial_prompt, "context": cultural_context})

validation_result = validate_prompt(layer1_prompt, context)

print("\nLayer 2 Validation Output:")
print(validation_result)