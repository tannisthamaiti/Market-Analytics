from fastapi import FastAPI, WebSocket
import uvicorn
import shutil
from smolagents import tool, Tool,HfApiModel, CodeAgent
from .map_agents import agent, task, task2, task3
import pandas as pd
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from fastapi.responses import JSONResponse
import os

# Load Janus model and processor
MODEL_PATH = "deepseek-ai/Janus-Pro-1B"
vl_chat_processor = VLChatProcessor.from_pretrained(MODEL_PATH)
tokenizer = vl_chat_processor.tokenizer
vl_gpt = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
vl_gpt = vl_gpt.to(torch.bfloat16).eval()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:8080"] for tighter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("👋 Connected. Please send folder path.")

    folder_path = await websocket.receive_text()
    file_path = "/app/well_files/Well data.CSV"
    await websocket.send_text(f"📂 Running agent on folder: {file_path}")
    print(file_path)

    try:
        result = agent.run(task, additional_args={"files": file_path})
        
        # Extract column names and file name
        info = result[0]
        file_name = info["file_name"]
        lat_col = info["lat"]
        lon_col = info["lon"]
        name_col = info.get("well_name")
        await websocket.send_text(f"📄 Detected columns - Lat: {lat_col}, Lon: {lon_col}, Name: {name_col}")
        await websocket.send_text("📝 Confirm: reply 'yes' if correct, or 'no' to provide a corrected file name.")

        confirmation = await websocket.receive_text()

        if confirmation.strip().lower() == "yes":
            await websocket.send_text("✅ Thanks! Confirmation received. Will plot the well locations now!")
            # Step 2: Load the data directly here
            
            df = pd.read_csv(file_path, on_bad_lines='skip')
            agent.run(task3, additional_args={
                "data": df,
                "lat_col": lat_col,
                "lon_col": lon_col,
            })
            await websocket.send_text("✅ Click to visualize plot!")
        
        elif confirmation.strip().lower() == "no":
            await websocket.send_text("🔍 Please provide the corrected file name (with well locations).")
            corrected_file = await websocket.receive_text()
            await websocket.send_text(f"✅ Using corrected file: {corrected_file}")
            # 👉 Optional: run agent again with corrected file if needed
            corrected_result = agent.run(task2, additional_args={"files": corrected_file})
            await websocket.send_text(f"🎉 Result with corrected file: {corrected_result}")
        
        else:
            await websocket.send_text("❓ Invalid response. Please reply 'yes' or 'no'.")

    except Exception as e:
        await websocket.send_text(f"⚠️ Error: {str(e)}")
@app.get("/output/plot.html")
async def serve_plot():
    return FileResponse(
        path="/app/output/plot1.html",  # Adjust if different
        media_type="text/html"
    )

@app.post("/ask")
async def ask_image_question(image: UploadFile = File(...), question: str = Form(...)):
    # Save image to a temporary file
    try:
        temp_path = f"/tmp/{image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Build conversation
        conversation = [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{question}", "images": [temp_path]},
            {"role": "<|Assistant|>", "content": ""}
        ]

        # Load and process image
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(vl_gpt.device)

        # Generate response
        inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
        )

        # Decode and return response
        answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return JSONResponse(content={"answer": answer})
    except Exception as e:
        print("❌ Internal Error:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)

UPLOAD_DIR = "/app/well_files"  # Mounted via docker-compose

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "status": "Uploaded successfully"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
