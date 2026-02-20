
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from model import ChromatinCNN
from pyfaidx import Fasta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

hg19 = Fasta("hg19.fa")

model = ChromatinCNN()
model.load_state_dict(torch.load("promoter_split_model.pth", map_location=torch.device("cpu")))
model.eval()

class CoordInput(BaseModel):
    promoter_chr: str
    promoter_start: int
    promoter_end: int
    interactor_chr: str
    interactor_start: int
    interactor_end: int

def one_hot_encode(sequence):
    mapping = {"A":[1,0,0,0],"C":[0,1,0,0],"G":[0,0,1,0],"T":[0,0,0,1]}
    sequence = sequence.upper()
    encoded = [mapping.get(base,[0,0,0,0]) for base in sequence]
    arr = np.array(encoded).T
    tensor = torch.tensor(arr,dtype=torch.float32).unsqueeze(0)
    return tensor

@app.get("/")
def health():
    return {"status":"Backend running"}

@app.post("/predict")
def predict(data: CoordInput):
    try:
        prom_seq = hg19[data.promoter_chr][data.promoter_start:data.promoter_end].seq
        int_seq = hg19[data.interactor_chr][data.interactor_start:data.interactor_end].seq
    except:
        return {"error":"Invalid coordinates"}

    promoter_tensor = one_hot_encode(prom_seq)
    interactor_tensor = one_hot_encode(int_seq)

    with torch.no_grad():
        output = model(promoter_tensor, interactor_tensor)
        prob = torch.sigmoid(output).item()

    if prob >= 0.7:
        label = "High likelihood"
    elif prob >= 0.4:
        label = "Medium likelihood"
    else:
        label = "Low likelihood"

    return {"probability":prob,"prediction":label}
