"""nanochat Task for reinforcement learning on OOA flows (copy to nanochat/tasks/ooa_rl.py).
Rows: JSONL {"messages":[{"role":"user",...},{"role":"assistant","content":<gold flow json>}]}.
nanochat's chat_rl.py samples several answers per request and calls reward(); answers the verifier scores higher than
the group average are reinforced (REINFORCE with advantage = reward - mean). evaluate() = fully solved (pass@k).
"""
import json, sys
from tasks.common import Task


class OOAFlowRL(Task):
    def __init__(self, path, ooa_dir="/content/ooa", **kwargs):
        super().__init__(**kwargs)
        if ooa_dir not in sys.path:
            sys.path.insert(0, ooa_dir)
        import ooa_reward
        self.R = ooa_reward
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        return {"messages": self.rows[index]["messages"]}

    def _score(self, conversation, response):
        gold = self.R.gold_actions(conversation["messages"][-1]["content"])
        return self.R.score(response, gold)

    def evaluate(self, conversation, assistant_response):
        return int(self._score(conversation, assistant_response)["solved"])

    def reward(self, conversation, assistant_response):
        return float(self._score(conversation, assistant_response)["total"])
