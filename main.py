from fastapi import FastAPI

from router import router

app = FastAPI(
    title="Olist Business Decision Copilot",
    description="Grounded policy answers and safe, read-only Olist analytics.",
    version="1.0.0",
)
app.include_router(router)


@app.get("/")
def root():
    return {
        "service": "Olist Business Decision Copilot",
        "docs": "/docs",
        "setup": "python -m scripts.run_all --setup",
    }