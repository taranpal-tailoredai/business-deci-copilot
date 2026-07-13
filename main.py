from fastapi import FastAPI

app =  FastAPI()

@app.get("/")
async def root():
    return {"mssg":"welcome to tailored ai"}