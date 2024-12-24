from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = FastAPI()

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_sakhi_model"  # Path where your model is saved
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("input", "")

    if not user_input:
        return {"response": "Please provide an input."}

    response = text_generator(user_input, max_length=150, num_return_sequences=1)
    return {"response": response[0]["generated_text"]}
