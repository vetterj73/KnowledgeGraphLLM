# app/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from azure.core.exceptions import ResourceExistsError
from .models import UserPrompt
from .azure_client import AzureBlobClient
#  from .openai_client import get_completion
from .config import settings
import networkx as nx
import pickle
import faiss
import gzip
import io

app = FastAPI(default_response_class=JSONResponse)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), azure_client: AzureBlobClient = Depends()):
    try:
        data = await file.read()
        await azure_client.upload_file(file.filename, data)
        return {"message": f"Successfully uploaded {file.filename}"}
    except ResourceExistsError:
        raise HTTPException(status_code=409, detail="File already exists.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_prompt(user_prompt: UserPrompt, azure_client: AzureBlobClient = Depends()):
    try:
        # graph_data: bytes = await azure_client.download_blob(settings.GRAPH_PATH)
        # with gzip.open(io.BytesIO(graph_data), 'rb') as f:
        #     graph_data_decompressed: bytes = f.read()
        # graph = pickle.loads(graph_data_decompressed)

        # index_data: bytes = await azure_client.download_blob(settings.INDEX_PATH)
        # index = faiss.read_index(io.BytesIO(index_data))

        # id_dict_data = await azure_client.download_blob(settings.MAPPING_PATH)
        # with open(io.BytesIO(id_dict_data), 'rb') as f:
        #     id_to_entity: dict = pickle.load(f)
        # blobs = await azure_client.list_blobs()
        # contents = []
        # for blob in blobs:
        #     data = await azure_client.download_blob(blob.name)
        #     contents.append(data.decode('utf-8'))
        # combined_content = "\n".join(contents)
        # prompt = f"{combined_content}\n\nUser Prompt: {user_prompt.prompt}"
        # response_text = await get_completion(prompt)
        return {"response": "response_text"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
