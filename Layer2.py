import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

os.environ["GROQ_API_KEY"] = "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"

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)

def validate_prompt(initial_prompt, cultural_context):
    template = ChatPromptTemplate.from_template(
        "You are a strict Historical Accuracy Critic Agent. Analyze the visual prompt for Flux. "
        "Prompt: {prompt}\n"
        "Context: {context}\n"
        "Check by criteria: did materials and tools exist in the specified era, are they characteristic of this culture. "
        "If there are anachronisms, correct the prompt for historical accuracy while preserving visual expressiveness. "
        "Respond strictly in the following format:\n"
        "Analysis: [Brief error analysis]\n"
        "Refined prompt: [Final corrected prompt in English]\n"
        "Do not write any additional notes, introductions, or conclusions. Only the specified format."
    )
    chain = template | llm | StrOutputParser()
    return chain.invoke({"prompt": initial_prompt, "context": cultural_context})

bad_prompt = "A traditional Russian craftsman painting a wooden bowl with a modern metal airbrush, wearing a synthetic apron, hyperrealistic."
context = "Russian Khokhloma, 17th century, Nizhny Novgorod province. Wooden blanks, squirrel hair brushes, and tempera paints were used."

print("INITIAL PROMPT:")
print(bad_prompt)

print("\nCRITIC VALIDATOR:")
result = validate_prompt(bad_prompt, context)
print(result)