from fastapi import FastAPI
from instrument_classification_module import InstrumentClassification

print("loading instrument model")
ic = InstrumentClassification()

app = FastAPI()

@app.get("/")
async def root():
    return { "message": 'It works! /docs for documentation' }

@app.get("/instrument_type")
# async def instrument_type():
#     return { "instrument_types": ["clarinet", "trumpet"] }

async def instrument_type():
    print("instrument types",ic.getInstrumentTypes())
    return { "instrument_types": ic.getInstrumentTypes() }


@app.get("/instrument_type/detect")
async def detect_instrument_type(filename:str):
    predicted_instrument = ic.classify(filename)
    return {
        "predicted_instruments": predicted_instrument
    }


# @app.get("/instrument_type/{instrument_name}")
# async def getInstrument_type(instrument_name):
#     return { "instrument_type": instrument_name }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)