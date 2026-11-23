"""nanochat Task for reinforcement learning on OOA requests in the served format (copy to nanochat/tasks/ooa_rl.py).
Rows (rl_setup.prepare): {"messages": [{"role": "user", "content": <request + retrieval context, as served>},
{"role": "assistant", "content": <gold answer: plan + compact flow, or "Question: ...">}], "gold": {...}}.
The chat_rl.py patched by rl_setup.patch_chat_rl samples several answers per request and calls score(); answers that
score above the group's mean are reinforced (REINFORCE, advantage = reward - mean). The reward is rl_setup.reward: the
platform checker on the expanded flow (see rl_setup.REWARD_DOC). The OOA modules are read from env OOA_DIR
(default /content/ooa); the catalog and schema from OOA_CATALOG / OOA_SCHEMA.
"""
import json
import os
import sys

from tasks.common import Task


class OOARL(Task):
    def __init__(self, path, ooa_dir=None, **kwargs):
        super().__init__(**kwargs)
        ooa_dir = ooa_dir or os.environ.get("OOA_DIR", "/content/ooa")
        if ooa_dir not in sys.path:
            sys.path.insert(0, ooa_dir)
        import rl_setup
        self.S = rl_setup
        rl_setup.configure()
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        r = self.rows[index]
        return {"messages": r["messages"], "gold": r["gold"]}

    def score(self, conversation, assistant_response, ended=True, tokens=0):
        return self.S.reward(conversation["gold"], assistant_response, ended=ended, tokens=tokens)

    def evaluate(self, conversation, assistant_response):
        return int(self.score(conversation, assistant_response)["solved"])

    def reward(self, conversation, assistant_response, ended=True, tokens=0):
        return float(self.score(conversation, assistant_response, ended, tokens)["total"])
