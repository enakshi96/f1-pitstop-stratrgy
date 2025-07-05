# main.py

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi import Query
from strategy import analyze_strategy
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

app = FastAPI(
    title="F1 Pit Strategy Recommender",
    description="Predicts and analyzes race strategies using FastF1 and Groq insights",
    version="2.0"
)
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/strategy_result", response_class=HTMLResponse)
def strategy_result(request: Request, year: int, race: str, driver: str):
    result = analyze_strategy(year, race, driver)
    if "error" in result:
        return templates.TemplateResponse("index.html", {"request": request, "error": result["error"]})
    return templates.TemplateResponse("index.html", {"request": request, "data": result})

@app.get("/strategy", summary="Get pit strategy and insights")
def get_strategy(
    year: int = Query(..., description="Race year (e.g. 2023)"),
    race: str = Query(..., description="Race name (e.g. 'British Grand Prix')"),
    driver: str = Query(..., description="Driver code (e.g. 'VER')")
):
    result = analyze_strategy(year, race, driver)
    
    # Add basic error handling to propagate API-friendly errors
    if "error" in result:
        return {"status": "error", "message": result["error"]}
    
    return {
        "status": "success",
        "data": result
    }
