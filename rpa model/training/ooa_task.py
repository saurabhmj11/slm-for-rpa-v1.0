"""nanochat Task that serves OOA dialogues from a JSONL file.

Copy into nanochat/tasks/ (or add to PYTHONPATH) and use in scripts/chat_sft.py's train_tasks list, e.g.
    from tasks.ooa import OOADialogues
    train_tasks = [OOADialogues("data/dialogues/train.jsonl"), ...]
Each JSONL line: {"messages": [{"role": "user"|"assistant", "content": str}, ...]}
 - optional first message with role "system" (nanochat merges it into the first user turn)
 - roles must alternate user, assistant, ...; assistant content is the minified Flow JSON or a short reply
Format checked against nanochat tasks/smoltalk.py and Tokenizer.render_conversation (commit 92d63d4).
"""
import json
from tasks.common import Task


class OOADialogues(Task):
    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        with open(path, encoding="utf-8") as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        messages = self.rows[index]["messages"]
        rest = messages[1:] if messages[0]["role"] == "system" else messages
        assert len(rest) >= 2
        for i, m in enumerate(rest):
            assert m["role"] == ("user" if i % 2 == 0 else "assistant")
            assert isinstance(m["content"], str)
        return {"messages": messages}
