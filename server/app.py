import sys
import os
from pathlib import Path

# Ensure the project root is on sys.path so `env` module can be imported
# regardless of whether we're run as `server.app` or directly.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI, HTTPException
from env import CustomerSupportEnv, Action, Observation, Reward, State
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(
    title="OpenEnv Customer Support Triage",
    version="1.0.0",
)

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
            info=info,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=State)
def get_state():
    return system_env.state()


def main():
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
