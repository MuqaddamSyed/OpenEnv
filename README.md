---
title: Customer Support Agent Simulation
emoji: 📞
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags: [openenv, reinforcement-learning, text-generation]
---
# Customer Support Agent - OpenEnv Simulation

## Environment Description & Motivation
This repository provides a custom OpenEnv environment simulating real-world **Customer Support Triage and Resolution**. It requires the agent to handle realistic service ticket queries, retrieve database information, and act on user issues. Support ticket handling provides a multi-step dynamic process, which reflects the daily tasks that customer satisfaction departments experience. It requires analytical, conversational, and decision-making capabilities from an agent.

## Action and Observation Space Definitions
### Observation Space (`env.Observation`)
- `ticket_id` (str): Unique representation of the current active support ticket.
- `ticket_content` (str): Raw message sent by the customer.
- `system_messages` (List[str]): Immediate feedback and responses from the environment based on previous agent actions.
- `knowledge_base_articles` (Dict[str, str]): Environment rules representing mock Support Policies.
- `customer_db_record` (Optional[Dict]): Returned specific user records upon successful DB query.
- `task_difficulty` (str): Indicator of current environment complexity (`easy`, `medium`, `hard`).

### Action Space (`env.Action`)
Agent must return standard **JSON actions**, wrapped in `Action` typed model:
- `tool` (str): The specific tool to invoke (`categorize`, `query_db`, `refund`, `reply`, `submit`).
- `arguments` (Dict): Parameter arguments required to formulate action accurately.

## Tasks and Difficulty
| Task | Difficulty | Objective | Grader Method |
|---|---|---|---|
| **Ticket Classification** | `easy` | Agent reads a cancellation user request and accurately issues `categorize` | Binary grader correctly validates category (`1.0`), incorrect gets (`0.0`). |
| **Policy Verification** | `medium` | Agent navigates Refund check. Queries DB (`query_db`) and checks policy constraints vs user days, issuing a refund (`refund`). | Grader is partial: awards DB execution (`+0.2`), and full logical resolution (`+0.8`). |
| **Full Escalation Resolution** | `hard` | Furious delayed customer. Agent queries DB (`query_db`), apologizes (`reply`), refunds 400 (`refund`), and issues a travel voucher. | Grader evaluates multi-step sequence progression: DB querying (`0.1`), polite reply (`0.3`), refund issued (`0.3`), and voucher assignment (`0.3`). |

## Setup & Usage Instructions
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your environment variables (Required for inference):
   ```bash
   export MODEL_NAME="gpt-4"
   export API_BASE_URL="https://api.openai.com/v1"
   export OPENAI_API_KEY="sk-..." # Or export HF_TOKEN="..."
   ```
4. Run inference locally:
   ```bash
   python inference.py
   ```

### Docker
To run inside a containerized setup compliant with HF Spaces:
```bash
docker build -t openenv_support .
docker run --env OPENAI_API_KEY=$OPENAI_API_KEY openenv_support
```

## Baseline Model Scores
Tested against **`gpt-4-turbo`**:
- **Easy Task**: 1.0 / 1.0
- **Medium Task**: 1.0 / 1.0
- **Hard Task**: 1.0 / 1.0

The baseline accurately demonstrates deterministic environment execution through multi-step capabilities with meaningful partial reinforcement signals in case of failure.
