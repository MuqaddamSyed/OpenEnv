from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Tuple

class Observation(BaseModel):
    ticket_id: str
    ticket_content: str
    system_messages: List[str]
    knowledge_base_articles: Dict[str, str]
    customer_db_record: Optional[Dict[str, Any]] = None
    task_difficulty: str

class Action(BaseModel):
    tool: str  # Tools: 'categorize', 'query_db', 'refund', 'reply', 'submit'
    arguments: Dict[str, Any]

class Reward(BaseModel):
    score: float

class State(BaseModel):
    task_id: str
    step_count: int
    resolved: bool
    final_score: float
    action_history: List[str]
    internal_state: Dict[str, Any]

class CustomerSupportEnv:
    def __init__(self, task_difficulty: str = "easy"):
        self.task_difficulty = task_difficulty
        self.reset()
        
    def reset(self) -> Observation:
        self.step_count = 0
        self.resolved = False
        self.final_score = 0.0
        self.action_history = []
        self.system_messages = ["Environment initialized. Awaiting action."]
        self.internal_state = {}
        
        # Setup specific task data based on difficulty
        if self.task_difficulty == "easy":
            self.ticket_id = "T-1001"
            self.ticket_content = "Hi, I would like to cancel my premium subscription. It is too expensive."
            self.knowledge_base = {
                "categories": "Possible categories: 'cancellation', 'technical', 'billing', 'general'"
            }
            self.customer_db = None
        elif self.task_difficulty == "medium":
            self.ticket_id = "T-2002"
            self.ticket_content = "Hello, I returned my defective shoes (Order O123) last week but haven't received my refund yet."
            self.knowledge_base = {
                "refund_policy": "Refunds are issued 5-7 business days after the return is processed. If more than 7 days have passed, issue a full refund."
            }
            self.customer_db = {"O123": {"days_since_return": 8, "total_amount": 120.0}}
            self.internal_state = {"db_queried": False, "refund_issued": False}
        elif self.task_difficulty == "hard":
            self.ticket_id = "T-3003"
            self.ticket_content = "This is OUTRAGEOUS! My flight F999 was delayed 6 hours and I missed my connection. Booking B456. DO SOMETHING AND COMPENSATE ME NOW!"
            self.knowledge_base = {
                "delay_policy": "Flight delays >4 hours get full refund plus $100 travel voucher. Must politely apologize, issue the refund, and optionally reply to the user."
            }
            self.customer_db = {"B456": {"flight": "F999", "delay_hours": 6, "total_amount": 400.0}}
            self.internal_state = {"refund_issued": False, "voucher_issued": False, "replied_politely": False, "apology_keywords": ["sorry", "apologize", "apologies"]}
        else:
            raise ValueError(f"Unknown task difficulty: {self.task_difficulty}")

        return self._get_obs()

    def _get_obs(self) -> Observation:
        return Observation(
            ticket_id=self.ticket_id,
            ticket_content=self.ticket_content,
            system_messages=self.system_messages,
            knowledge_base_articles=self.knowledge_base,
            customer_db_record=self.internal_state.get("current_db_record", None),
            task_difficulty=self.task_difficulty
        )

    def state(self) -> State:
        return State(
            task_id=self.task_difficulty,
            step_count=self.step_count,
            resolved=self.resolved,
            final_score=self.final_score,
            action_history=self.action_history,
            internal_state=self.internal_state
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self.resolved:
            return self._get_obs(), Reward(score=self.final_score), True, {}
            
        self.step_count += 1
        self.action_history.append(action.tool)
        self.system_messages = []
        reward_value = 0.0
        done = False
        
        # Penalize for infinite loops / taking too long
        if self.step_count > 10:
            self.resolved = True
            self.final_score = 0.0
            self.system_messages.append("Max steps reached. Failure.")
            return self._get_obs(), Reward(score=0.0), True, {"error": "max_steps"}

        tool = action.tool
        args = action.arguments

        # Graceful handling of malformed actions
        if tool not in ["categorize", "query_db", "refund", "reply", "submit"]:
            self.system_messages.append(f"Invalid tool: {tool}")
            reward_value = -0.1 # Small penalty for bad formatting
        
        else:
            # Grader Logic & Environment Transitions
            if self.task_difficulty == "easy":
                if tool == "categorize":
                    category = args.get("category", "").lower()
                    if category == "cancellation":
                        reward_value = 1.0
                        self.final_score = 1.0
                        done = True
                        self.system_messages.append("Correct category selected. Task complete.")
                    else:
                        reward_value = 0.0
                        self.final_score = 0.0
                        done = True
                        self.system_messages.append(f"Wrong category: {category}. Task complete.")
                else:
                    self.system_messages.append("Invalid action for this task. You only need to categorize.")

            elif self.task_difficulty == "medium":
                if tool == "query_db":
                    order_id = args.get("order_id", "")
                    if order_id in self.customer_db:
                        self.internal_state["current_db_record"] = self.customer_db[order_id]
                        self.internal_state["db_queried"] = True
                        self.system_messages.append("DB queried successfully.")
                        reward_value = 0.2  # Partial reward for investigating
                    else:
                        self.system_messages.append("Order ID not found.")
                elif tool == "refund":
                    order_id = args.get("order_id", "")
                    amount = args.get("amount", 0.0)
                    if not self.internal_state.get("db_queried"):
                        self.system_messages.append("Cannot issue refund without querying DB first.")
                        reward_value = -0.1
                    else:
                        if order_id == "O123" and amount == 120.0:
                            self.internal_state["refund_issued"] = True
                            self.system_messages.append("Refund issued correctly.")
                            reward_value = 0.8 # Brings total to 1.0 if queried
                        else:
                            self.system_messages.append("Incorrect refund amount or order.")
                elif tool == "submit":
                    if self.internal_state.get("refund_issued"):
                        self.final_score = min(1.0, 0.2 + 0.8)
                    else:
                        self.final_score = 0.0
                    reward_value = self.final_score
                    done = True
                    self.system_messages.append("Task medium submitted.")

            elif self.task_difficulty == "hard":
                if tool == "query_db":
                    booking = args.get("booking_id", args.get("order_id", ""))
                    if booking in self.customer_db:
                        self.internal_state["current_db_record"] = self.customer_db[booking]
                        self.system_messages.append("DB queried successfully.")
                        reward_value = 0.1
                    else:
                        self.system_messages.append("Booking ID not found in DB.")
                elif tool == "refund":
                    amount = args.get("amount", 0.0)
                    voucher = args.get("voucher", False)
                    # Policy expects total amount + $100 voucher
                    if amount == 400.0:
                        self.internal_state["refund_issued"] = True
                        reward_value += 0.3
                        self.system_messages.append("Refund correctly issued.")
                    if voucher:
                        self.internal_state["voucher_issued"] = True
                        reward_value += 0.3
                        self.system_messages.append("Voucher correctly issued.")
                elif tool == "reply":
                    content = args.get("message", "").lower()
                    if any(word in content for word in self.internal_state["apology_keywords"]):
                        self.internal_state["replied_politely"] = True
                        reward_value += 0.3
                        self.system_messages.append("Replied politely with apology.")
                    else:
                        self.system_messages.append("Reply lacks an apology.")
                elif tool == "submit":
                    score = 0.0
                    if self.internal_state.get("current_db_record"): score += 0.1
                    if self.internal_state.get("refund_issued"): score += 0.3
                    if self.internal_state.get("voucher_issued"): score += 0.3
                    if self.internal_state.get("replied_politely"): score += 0.3
                    self.final_score = min(1.0, score)
                    reward_value = self.final_score
                    done = True
                    self.system_messages.append("Task hard submitted.")

        self.resolved = done
        return self._get_obs(), Reward(score=reward_value), done, {"action_history": self.action_history}
