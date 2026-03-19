from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from database import GiftDatabase, ShowcaseDatabase
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

db = GiftDatabase(json_path='gifts.json')
showcase_db = ShowcaseDatabase(json_path='showcases.json')

class ShowcaseSaveRequest(BaseModel):
    user_id: int
    name: Optional[str] = None
    slots: List[dict]

@app.get("/search")
async def search(q: str = Query(None, min_length=2)):
    if not q:
        return db.gifts[:20]
    return db.search_gifts(q, top_k=20)

@app.post("/save_showcase")
async def save_showcase(req: ShowcaseSaveRequest):
    result = showcase_db.save_showcase(req.user_id, req.name, req.slots)
    return result

@app.get("/get_showcases")
async def get_showcases(user_id: int):
    return showcase_db.get_user_showcases(user_id)

@app.get("/stats")
async def stats():
    return {"total_models": len(db.gifts), "total_showcases": len(showcase_db.showcases)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
