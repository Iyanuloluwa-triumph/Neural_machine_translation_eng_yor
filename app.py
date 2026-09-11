import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

app = FastAPI()

# Enable CORS so your JS frontend can communicate with this API smoothly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Define paths and base model
base_model_id = "facebook/nllb-200-distilled-600M"
adapter_path = "./nllb_yoruba_lora_final"  # The folder with your saved adapter

# 2. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, src_lang="eng_Latn", tgt_lang="yor_Latn"
)

# 3. Load Base Model and merge with the LoRA Adapter
print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Move to GPU for fast generation (if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Define request body schema using Pydantic
class TranslationRequest(BaseModel):
    text: str

@app.post("/translate")  # FIX 1: Added missing 'app.' prefix
async def translate(request: TranslationRequest):
    text = request.text.strip()  # FIX 2: FastAPI automatically parses JSON into the Pydantic model
    
    if not text:
        return {"error": "No text provided for translation."}

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Grab the specific ID for Yoruba to force NLLB language routing
    yoruba_bos_id = tokenizer.convert_tokens_to_ids("yor_Latn")

    # Generate translation using the model
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            forced_bos_token_id=yoruba_bos_id,  # FIX 3: Crucial for NLLB target language routing
            max_length=128,
            num_beams=4,
            repetition_penalty=1.2
        )

    # Decode the generated tokens to get the translated text
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {"translated_text": translated_text}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)  # FIX 4: Corrected IP typo from '127.0.1' to '127.0.0.1'