from fastapi import FastAPI, HTTPException
from env import CustomerSupportEnv, Action, Observation, Reward, State
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

# Global environment instance for the Space API
system_env = CustomerSupportEnv()

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]

@app.get("/")
def read_root():
    return {"status": "ok", "message": "OpenEnv Customer Support Engine Running"}

@app.post("/reset", response_model=Observation)
def reset_env():
    return system_env.reset()

@app.post("/step", response_model=StepResponse)
def step_env(action: Action):
    try:
        obs, reward, done, info = system_env.step(action)
        return StepResponse(
            observation=obs,
            reward=reward,
            done=done,
            info=info
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state", response_model=State)
def get_state():
    return system_env.state()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
