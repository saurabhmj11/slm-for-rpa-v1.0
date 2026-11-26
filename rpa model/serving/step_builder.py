"""The step-by-step builder: a long request is cut into steps, each step is built on its own as a short flow
(the model is good at 1-3 step flows), checked, and the pieces are joined into one flow.

    request -> steps (split_steps: sentences / clauses / connectives; loops and ifs found from their words)
            -> per step: a plain request of its own (pronouns and "each one" made concrete: "Read the total for
               the invoice PDF"), retrieval for the step alone (Router.prompt on the step's text, the apps the
               whole request names searched with the step's words) + the variables known for it
            -> every step at once: the model's short answers (best-of-N at temperatures), ONE batched generation
               for all steps, each checked (Router.verify) and the best checked piece kept (fragment_score)
            -> loops built by code over the newest collection before them (a table -> to_records -> core/loop)
            -> the failed steps asked again in one more batch, with their problems and the variables the steps
               before them made
            -> join: a step inside a loop / if goes into that container's body ("otherwise" into its else),
               ids made unique, variables linked (ooa_format.link_part_variables), then repair + verify of the whole
            -> the router keeps whichever of this and the single-pass answer verifies and covers more of the request

Only the rules here decide the structure (which step is a loop, what goes inside it); the model decides the actions
of each step. Fast mode never uses this."""
import json
import re
import time

import ooa_format as F

# -- splitting a request into steps ------------------------------------------------------------------------------
# Verbs that start a step ("..., save the PDF attachments, mark each email read"). A comma or "and" before one of these
# starts a new step; an "and" between nouns ("bug, crash or error", "name and date") does not.
STEP_VERBS = set("""
read write save send email e-mail mail copy move delete remove rename download upload open close extract filter sort
merge split zip unzip compress convert print create make add update insert append fill click type search find check
compare calculate compute sum count export import fetch get post call run start stop attach reply forward archive
backup notify alert schedule scan parse translate summarize summarise resize crop rotate encrypt decrypt sign share
sync publish submit approve reject flag mark list load pull push put keep take build lock unlock protect unprotect
log ping wait cancel resend change set book ask have generate draft collect go loop look upload store record track
tag assign close notify message text tell remind verify validate match clean trim replace format fit bold autofit
delete drop dedupe deduplicate combine join group pivot total publish refresh restart stop terminate reboot query
fetch submit raise open escalate file transfer kick sign-in sign login log-in authenticate connect disconnect revoke
print ocr describe classify categorize categorise label analyse analyze review work schedule
""".split())
# a step that ends the loop it follows: a summary of all of it ("then email the team how many ...")
_AFTER_LOOP = re.compile(r"\b(?:how many|the (?:list|summary|totals?|count|report|results?)\b|a summary|in total|"
                         r"at the end|finally|afterwards|when (?:it is |it's )?(?:all )?done|once (?:all|done)|"
                         r"everything passed|all of them)", re.I)
_CTX = r"(?:(?:in|on|from|at|inside|within|using)\s+[^,]{1,60},\s*)?"
_LOOP_HEAD = re.compile(r"^" + _CTX + r"(?:and\s+)?(?:then\s+)?(?:for\s+(?:each|every|all)|go(?:ing)?\s+through|loop\s+(?:over|through)|"
                        r"process\s+(?:each|every))\b", re.I)
_EACH_INSIDE = re.compile(r"\b(?:each|every)\s+(?!evening|morning|day|night|week|month|hour|monday|tuesday|wednesday|"
                          r"thursday|friday|saturday|sunday|weekday|year|quarter|time)\w+", re.I)
_IF_HEAD = re.compile(r"^" + _CTX + r"(?:and\s+)?(?:then\s+)?(?:only\s+)?(?:if|when|in case|unless)\b", re.I)
_ELSE_HEAD = re.compile(r"^(?:and\s+)?(?:otherwise|else|if not|or else|if no\b|if there are none|if nothing|"
                        r"if none)", re.I)
_SCHEDULE = re.compile(r"^(?:every|each)\s+(?:evening|morning|day|night|week|month|hour|monday|tuesday|wednesday|"
                       r"thursday|friday|saturday|sunday|weekday|working day|business day)\b[,:]?\s*", re.I)
_LEAD = re.compile(r"^(?:and|then|also|after that|afterwards|next|finally|at the end|and then|,)\s+", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# hard boundaries inside a sentence: ";", "then", "after that", "also", "finally"
_HARD = re.compile(r"\s*;\s*|,?\s+(?:and\s+)?(?:then|after that|afterwards|also|finally|at the end)\s+(?=\w)", re.I)
_NOOP = re.compile(r"^(?:and\s+)?(?:move on|carry on|continue|skip it|go on|keep going)\W*$", re.I)
# a story's own subject before its verb ("First I open the website", "then we zip the copy"): dropped, so the step
# reads as a plain request ("open the website")
_FIRST_PERSON = re.compile(r"^(?:first(?:ly)?,?\s+|then\s+|next\s+|after that,?\s+)*(?:I|we)\s+(?:(?:will|would|have to|"
                           r"need to|must|should|usually|always|normally|just|also|then|first|manually)\s+)*(?=[a-z])",
                           re.I)
# a sentence about the request, not a step of it ("Can the bot do this?", "That's my month end")
_META = re.compile(r"^(?:(?:can|could|will|would|is|are|do|does)\s+(?:the\s+bot|you|it|this|that|a bot|the robot|we|i)\b|"
                   r"(?:that's|that is|this is|it's|it is)\s+(?:my|our|the|all|it)\b|(?:please\s+)?help\b|thanks?\b|"
                   r"thank you\b)", re.I)
_STORY = re.compile(r"(?:^|[.,;]\s*|\bthen\s+|\band\s+)(?:first\s+)?(?:I|we)\s+(?:\w+\s+)?[a-z]+", re.I)


def is_story(request):
    """A request told as what the person does ("First I open ..., then I log in ..."): 2+ clauses with I / we as the
    subject. Its single answer reads it well; auto does not send it to the step builder (wants)."""
    return len(_STORY.findall(str(request or ""))) >= 2
MIN_STEP_WORDS = 2
MAX_STEPS = 10


def _words(text):
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text)


def _starts_with_verb(text):
    w = _words(_LEAD.sub("", text.strip()))
    return bool(w) and w[0].lower() in STEP_VERBS


# words that are verbs and nouns: never where a loop head ends ("for each unread email in ...", "for every message")
NOUNISH = set("""email mail message file record post text call log list check mark flag query label tag review schedule
alert total count type sign book match order report request upload download transfer change update note
filter sort merge split scan""".split())
_SEP = r",(?!\d)\s*(?:and\s+)?|\s+and\s+"


def _soft_split(seg):
    """Cut one segment on ", <verb>" and " and <verb>" (and ", and"), never inside quotes or numbers (10,000).
    A first piece without any verb ("In the training tracker Excel, find ...") stays with the next one."""
    out, cur = [], ""
    for tok in re.split(r"(" + _SEP + r"|'[^']*'|\"[^\"]*\")", seg):
        if not tok:
            continue
        if re.fullmatch(_SEP, tok):
            out.append((cur, tok))
            cur = ""
            continue
        cur += tok
    out.append((cur, ""))
    pieces = []
    for text, sep in out:
        t = text.strip()
        starts = _starts_with_verb(t) or _LOOP_HEAD.match(t) or _IF_HEAD.match(t) or _ELSE_HEAD.match(t)
        lone = len(pieces) == 1 and not any(w.lower() in STEP_VERBS for w in _words(pieces[0][0]))
        if pieces and (not starts or lone):
            prev_text, prev_sep = pieces[-1]
            pieces[-1] = (prev_text + prev_sep + text, sep)
        else:
            pieces.append((text, sep))
    return [re.sub(r"\s+and$", "", p.strip()) for p, _ in pieces if p.strip()]


def _split_loop_head(text):
    """'for each rejected payment set its status ...' -> ('for each rejected payment', 'set its status ...');
    no verb after the head (or only one that is also a noun): (text, '') and the loop is built with its body."""
    m = _LOOP_HEAD.match(text)
    for x in re.finditer(r"[A-Za-z][A-Za-z'-]*", text):
        if x.start() < m.end() + 1:
            continue
        w = x.group(0).lower()
        if w in STEP_VERBS and w not in NOUNISH:
            return text[:x.start()].strip(" ,"), text[x.start():].strip(" ,")
    return text.strip(" ,"), ""


def _split_if_head(text):
    """'if it's over 10,000 forward it for approval' -> ("if it's over 10,000", 'forward it for approval');
    'if there are any' -> ('if there are any', '')."""
    m = _IF_HEAD.match(text)
    for x in re.finditer(r"[A-Za-z][A-Za-z'-]*", text):
        if x.start() < m.end() + 1:
            continue
        word = x.group(0).lower()
        before = text[:x.start()].rstrip()
        after = text[x.end():].lstrip()
        verb_like = re.match(r"(?:the|a|an|it|them|him|her|me|us|all|each|every|this|that|those|these)\b", after, re.I)
        if word in STEP_VERBS and (word not in NOUNISH or before.endswith(",") or verb_like):
            if re.search(r"\b(?:yes|no|ok|any|none|so|over|under|above|below|than|matches|match|passed|failed|ago|"
                         r"\d[\d,.%]*)$|,$", before, re.I):
                return before.strip(" ,"), text[x.start():].strip(" ,")
    return text.strip(" ,"), ""


_LOOP_WORDS = re.compile(r"^" + _CTX + r"(?:and\s+)?(?:then\s+)?(?:for\s+(?:each|every|all)(?:\s+of)?|go(?:ing)?\s+through|"
                         r"loop\s+(?:over|through)|process\s+(?:each|every)|each|every)\s*", re.I)
_FILTER = re.compile(r"\s+(?:that|which|whose|where|who|sent|older than|newer than|more than|with (?:a |an )?"
                     r"(?:status|amount|total|value|due date)|not\b)\b", re.I)
_PREP = re.compile(r"\s+(?:in|from|on|at|of|for|inside|under|within|into|to|with)\s+", re.I)
_REFERS = {"", "one", "ones", "of them", "them", "item", "items", "each", "entry", "entries"}
_FORMAT_WORDS = {"pdf", "pdfs", "file", "files", "csv", "document", "documents", "row", "rows", "record", "records",
                 "doc", "docs", "attachment", "attachments"}
_GENERIC_NOUN = {"one", "ones", "item", "items", "entry", "entries", "each", "them"}


def singular(word):
    w = str(word or "").lower().strip(" '\"")
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def plural(word):
    w = str(word or "")
    if not w or w.lower().endswith("s") and not w.lower().endswith("ss"):
        return w
    if w.isupper():
        return w + "s"                               # PDF -> PDFs
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    if re.search(r"(?:s|x|z|ch|sh)$", w):
        return w + "es"
    return w + "s"


def plural_phrase(what, noun):
    """'new invoice PDF in the inbox folder' (noun 'new invoice PDF') -> 'new invoice PDFs in the inbox folder': the
    loop's source step asks for all of them."""
    noun = str(noun or "").strip()
    if not noun or not what.startswith(noun):
        return what
    words = noun.split()
    return " ".join(words[:-1] + [plural(words[-1])]) + what[len(noun):]


def item_name(noun):
    """A loop item's variable name from its noun: 'invoice PDF' -> 'invoice', 'payments' -> 'payment'."""
    words = [w for w in re.findall(r"[A-Za-z]+", str(noun or "")) if w.lower() not in ("the", "a", "an", "all", "new",
                                                                                         "unread", "open", "our")]
    if not words:
        return "item"
    last = words[-1].lower()
    if last in _FORMAT_WORDS and len(words) > 1:
        last = words[-2].lower()
    name = re.sub(r"[^a-z0-9_]", "", singular(last)) or "item"
    return "item" if name in _GENERIC_NOUN else name


def parse_loop_head(head):
    """'For each new invoice PDF in the inbox folder' -> {"what": 'new invoice PDF in the inbox folder',
    "noun": 'new invoice PDF', "filter": '', "context": ''}; 'for every message that mentions a bug' ->
    noun 'message', filter 'that mentions a bug'; 'On a DocuSign envelope, go through the recipients' -> context
    'On a DocuSign envelope'."""
    text = head.strip(" ,.")
    ctx = ""
    m = re.match(_CTX, text, re.I)
    if m and m.group(0):
        ctx, text = m.group(0).strip(" ,"), text[m.end():]
    m = _LOOP_WORDS.match(text)
    what = text[m.end():].strip(" ,") if m else text
    what = re.sub(r"^(?:the|all(?: the)?|of the)\s+", "", what, flags=re.I)
    f = _FILTER.search(" " + what)
    noun, filt = (what[:f.start()].strip(), what[f.start():].strip()) if f else (what, "")
    p = _PREP.search(" " + noun)
    core = noun[:p.start()].strip() if p else noun
    return {"what": what, "phrase": noun, "noun": core, "filter": filt, "context": ctx}


_THING_DROP = {"the", "a", "an", "all", "new", "unread", "open", "our", "my", "every", "each", "one", "ones", "of",
               "them", "only", "any"}


def thing_of(noun):
    """The loop item in words, for the steps inside the loop: 'new invoice PDF' -> 'invoice PDF', 'rejected payment'
    -> 'rejected payment', 'one' -> '' (unknown: the earlier step's collection names it)."""
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]*", str(noun or "")) if w.lower() not in _THING_DROP]
    for k in range(1, len(words)):            # "incidents assigned to ..." -> incidents
        if re.search(r"(?:ed|ing)$", words[k].lower()) and len(words[k]) > 4:
            words = words[:k]
            break
    if not words:
        return ""
    words[-1] = words[-1] if words[-1].isupper() else singular(words[-1])
    return " ".join(words[-2:])


def refers_back(info, earlier_text):
    """True when a loop goes over something an earlier step already got: 'for each one', 'for each' or a noun the
    earlier steps name ('for every message ...' after 'Read today's messages ...')."""
    noun = info["noun"].lower().strip()
    if noun in _REFERS:
        return True
    words = [singular(w) for w in re.findall(r"[a-z]+", noun) if w not in ("new", "unread", "open", "the")]
    head = item_name(noun)
    seen = {singular(w) for w in re.findall(r"[a-z]+", earlier_text.lower())}
    return bool(words) and (head in seen or words[-1] in seen) and not _PREP.search(" " + info["phrase"])


_FIND_ONE = re.compile(r"^(?:find|pick|get|take|select|look for|spot)\s+(?:out\s+)?the\s+(?:ones?|items?|records?|rows?|"
                       r"entries|entry)\s+(with|whose|where|that|which)\s+(.+)$", re.I)
_OR_THAT = re.compile(r"^(.+?),?\s+or\s+(?:else\s+)?(?:that|say that)\s+(.+)$", re.I)


def split_steps(request):
    """The request as steps to build, in order: [{"text", "kind": do|loop|if, "parent", "branch", ...}].
    parent: the index of the loop / if step whose body the step goes into (None: the top level); branch: "children"
    (a loop's body, an if's then) or "else".
    A loop step is built by code (Router side: build): the collection an earlier step made (its "refers" is True:
    "for each one", "for every message ..." after a step that read the messages), else a step that gets it (its
    "source" text: 'Get the new invoice PDF in the inbox folder'), then (a table) to_records and core/loop over it;
    its "item" names the loop item. A filter on a collection an earlier step made ("for every message that mentions
    a bug") is an if step inside the loop. An if step is built by the model as one piece: its condition and its first
    action ("If it's over 10,000, forward it ..."); the steps after it in the same clause go into its then, "otherwise
    ..." into its else.
    Rules: sentences, then hard boundaries (";", "then", "after that", "also", "finally"), then ", <verb>" /
    " and <verb>". A loop's body lasts until a step after a hard boundary that sums it all up ("; then email AP how
    many ..."); a hard boundary closes ifs. A step with its own "each" ("email each one a reminder") is a loop when
    none is open (closed at the next hard boundary). Returns [] for a request of fewer than 2 steps."""
    text = " ".join(str(request or "").split())
    text = _SCHEDULE.sub("", text)
    raw = []                      # (text, hard_boundary_before)
    for si, sent in enumerate(_SENTENCE.split(text)):
        segs = _HARD.split(sent.strip().rstrip("."))
        for gi, seg in enumerate(segs):
            for pi, piece in enumerate(_soft_split(seg)):
                raw.append((piece.strip(" ,."), (si > 0 or gi > 0) and pi == 0))
    steps, stack, branch = [], [], {}          # stack: open loops / ifs; branch: if index -> "children" | "else"

    def add(text, kind, **kw):
        parent = stack[-1] if stack else None
        steps.append({"text": text, "kind": kind, "parent": parent,
                      "branch": branch.get(parent, "children") if parent is not None else "children", **kw})
        return len(steps) - 1

    def add_if(piece, **kw):
        """An if step (its condition) and, when the clause says one, its first action inside it."""
        cond, body = _split_if_head(piece)
        stack.append(add(cond, "if", **kw))
        if body:
            add(body, "do")

    def add_loop(head, inline=False, piece=""):
        info = parse_loop_head(head)
        earlier = " ".join(s["text"] for s in steps)
        refers = refers_back(info, earlier)
        m = re.match(_CTX, piece, re.I) if piece else None
        ctx = info["context"] or (m.group(0).strip(" ,") if m else "")
        if inline and steps and not ctx and not _PREP.search(" " + info["phrase"]):
            refers = True                 # "OCR each piece" after "split it into documents": what the step before made
        item = item_name(info["noun"])
        source = ""
        if not refers:
            # the collection without its filter ("unread emails in the Bug Reports folder", not "... that have
            # screenshots"): a short request the model answers with the one list action
            what = info["phrase"] if not inline else info["noun"]
            what = plural_phrase(what, info["noun"])
            source = (ctx + ", list the " + what) if ctx else "List the " + what
        i = add(head, "loop", refers=refers, item=item, source=source, inline=inline, thing=thing_of(info["noun"]))
        stack.append(i)
        if refers and info["filter"]:
            noun = info["noun"] if info["noun"].lower() not in _REFERS else "item"
            filt = re.sub(r"^(?:that|which|who)\s+", "", info["filter"], flags=re.I)
            add_if(f"If the {noun} {filt}", filter=True)
        return i

    def add_find(piece):
        """Inside a loop, "find the one with the wrong address" picks items: an if on each item (True: added)."""
        m = _FIND_ONE.match(piece)
        loops = [k for k in stack if steps[k]["kind"] == "loop"]
        if not m or not loops:
            return False
        loop = steps[loops[-1]]
        noun = loop.get("thing") or loop.get("item") or "item"
        word, rest = m.group(1).lower(), m.group(2).strip(" ,")
        if word == "whose":
            cond = f"If the {noun}'s {rest}"
        else:
            cond = f"If the {noun} {'has' if word == 'with' else word} {rest}"
        stack.append(add(cond, "if", filter=True))
        return True

    for piece, hard in raw:
        piece = _LEAD.sub("", piece).strip(" ,.")
        if _META.match(piece) or piece.endswith("?"):
            continue
        piece = _FIRST_PERSON.sub("", piece).strip(" ,.")
        if not piece or _NOOP.match(piece):
            continue
        if len(_words(piece)) < MIN_STEP_WORDS and not _ELSE_HEAD.match(piece) and steps:
            steps[-1]["text"] += " " + piece
            continue
        if hard and stack:
            if _AFTER_LOOP.search(piece) and not _ELSE_HEAD.match(piece):
                stack = []
            else:
                # a hard boundary ("; then", a new sentence) closes the ifs and the loops a step's own "each" opened
                while stack and ((steps[stack[-1]]["kind"] == "if" and not steps[stack[-1]].get("filter"))
                                 or steps[stack[-1]].get("inline")):
                    if _ELSE_HEAD.match(piece) and steps[stack[-1]]["kind"] == "if":
                        break
                    stack.pop()
        last = steps[-1] if steps else None
        if _ELSE_HEAD.match(piece):
            ifs = [i for i in stack if steps[i]["kind"] == "if"]
            if ifs:
                stack = stack[:stack.index(ifs[-1]) + 1]
                branch[ifs[-1]] = "else"
                add(_ELSE_HEAD.sub("", piece).strip(" ,") or piece, "do")
                continue
            if not _IF_HEAD.match(piece):
                piece = _ELSE_HEAD.sub("", piece).strip(" ,") or piece
        if _LOOP_HEAD.match(piece):
            head, body = _split_loop_head(piece)
            add_loop(head)
            if body and not add_find(body):
                add(body, "do")
            continue
        if _IF_HEAD.match(piece):
            while stack and steps[stack[-1]]["kind"] == "if" and not steps[stack[-1]].get("filter"):
                stack.pop()
            add_if(piece)
            continue
        if add_find(piece):
            continue
        alt = _OR_THAT.match(piece)
        if alt and _TO_PERSON.match(piece) and steps:
            # "email QA lead how many failed, or that everything passed": an if, the message, else the other one
            first, other = alt.group(1).strip(" ,"), alt.group(2).strip(" ,")
            who = _REPHRASE_EMAIL.match(first)
            if who:
                while stack and steps[stack[-1]]["kind"] == "if" and not steps[stack[-1]].get("filter"):
                    stack.pop()
                i = add("if there are any", "if")
                stack.append(i)
                add(first, "do")
                branch[i] = "else"
                verb = first.split()[0]
                add(f"{verb} {who.group(1)} that {other}", "do")
                stack.pop()
                continue
        if not stack and _EACH_INSIDE.search(piece) and not _AFTER_LOOP.search(piece):
            m = _EACH_INSIDE.search(piece)
            # the each phrase up to the next verb ("each one", "every quote whose ... has passed")
            add_loop(m.group(0), inline=True, piece=piece)
        add(piece, "do")
    steps = steps[:MAX_STEPS]
    return steps if len([s for s in steps if s["kind"] == "do"]) >= 2 else []


# -- building ------------------------------------------------------------------------------------------------------
STEP_MAX_TOKENS = 450          # one step's answer: a short plan and a 1-4 step flow (about 250 tokens; more is a
                               # runaway answer that would hold the whole batch up)
LOOP_TEXT = "plain"            # how a step inside a loop is asked: "plain" (a sentence naming the item) or
                               # "with_current" ("With the current invoice: ...", the parts' wording)
STEP_TEMPERATURES = (0.2, 0.6, 0.9)   # the samples of one step (one batch): the usual answer and two freer ones
STEP_ROUNDS = 2                # a step none of whose samples passed is asked again once with its problems
STEP_APP_TOP = 2               # per app the step works in, its best actions for the step's words go in the context
MAX_SAME_ACTION = 2            # a piece with one action more often than this is a runaway answer, never used
SWAP_TOP = 6                   # per named app, its best actions for a step's words that an app swap may use
MAX_FAILED_SHARE = 0.34        # more failed steps than this share: no step-built flow
AGREE = 0.75                   # a checked piece's score per other sample of its step with the same actions
GUIDED_SAMPLE = True           # a step whose own words clearly point at actions (step_verb_ids strong: "keep the ones
                               # whose ...", "more than 7 days ago", "collect the links") gets one more sample whose
                               # answer starts with those actions as its plan (guided_prefix); the model writes the
                               # flow for it, and the checker and fragment_score judge it as any other sample
GUIDED_BONUS = 1.0             # a checked guided piece's extra score: its actions come from the step's own words
GUIDED_APP_BONUS = 0.5         # ... and a guided piece of the app a step names (app_pick)
SIMILAR_FIRST_ONLY = True      # only a step's first sample sees the similar examples; the freer ones do not (a short
                               # step request copies a similar example's whole outline too readily)
_CONTAINER = {"loop": "core/loop", "if": "core/if"}
_SETUP = re.compile(r"/(?:connect|authenticate|login|sign_in|open)$")
_COLLECTION = ("Table", "List")
_PRONOUN = re.compile(r"\b(it's|they're|it|its|them|they|their)\b", re.I)
_OBJ_STOP = re.compile(r"\s+(?:from|to|into|in|on|at|with|as|for|by|of|that|which|where|whose|tagged|named|using|"
                       r"via|so|if|when|and)\b.*$", re.I)


def object_of(text):
    """The thing a step works on, for 'it' / 'them' in the next step: 'Download the bank's payment status report
    from SFTP' -> "the bank's payment status report"; '' when there is none."""
    t = _LEAD.sub("", text.strip())
    look = re.match(r"(?:check|search|look in|look through|scan|go through)\s+.{1,60}?\s+for\s+(.*)$", t, re.I)
    if look:                                  # "Check the mailbox for unread emails from ..." -> the unread emails
        t = "get " + look.group(1)
    m = re.match(r"(?:" + _CTX + r")?[A-Za-z-]+\s+(?:out\s+|up\s+|on\s+)?(.*)$", t, re.I)
    rest = _OBJ_STOP.sub("", m.group(1)).strip(" ,") if m else ""
    if not rest or len(rest.split()) > 7 or _PRONOUN.fullmatch(rest):
        return ""
    if not re.match(r"(?:the|a|an|all|our|my|their|its|this|that|these|those|each|every)\b", rest, re.I):
        rest = "the " + rest
    return rest


_TO_PERSON = re.compile(r"(?:e-?mail|mail|notify|tell|remind|message|text|ping|post|reply|send)\b", re.I)


def subject_of(text):
    """The thing a condition talks about: "the invoice total is over 10,000" -> 'the invoice total'; '' when it
    starts with a pronoun or has no 'is / are / has ...'."""
    m = re.match(r"((?:the|a|an|its|their|our|any|all)\s+[^,]{1,60}?)\s+(?:is|are|was|were|has|have|had|does|do|"
                 r"can't|cannot|can|isn't|aren't|mentions?|matches?|exceeds?)\b", str(text or "").strip(), re.I)
    return m.group(1) if m else ""


def resolve_pronouns(text, thing):
    """'forward it for approval' + 'the invoice' -> 'forward the invoice for approval' (it's -> the invoice is,
    its / their -> the invoice's)."""
    if not thing:
        return text

    def sub(m):
        w = m.group(1).lower()
        if w == "it's":
            return thing + " is"
        if w == "they're":
            return thing + " are"
        if w in ("its", "their"):
            return thing + ("'" if thing.endswith("s") else "'s")
        if w == "they":
            return thing
        return thing
    return _PRONOUN.sub(sub, text)


def step_toolkits(request, built):
    """The apps a step works in: those the whole request names and those of the steps built so far (newest first)."""
    out = []
    for flow in reversed([b for b in built if b is not None]):
        for a in F.actions_of(flow):
            t = str(a).split("/")[0]
            if not t.startswith("core") and t not in out:
                out.append(t)
    return list(dict.fromkeys(F.named_apps(request) + out))


def step_hints(index, cat, text, toolkits, top=STEP_APP_TOP):
    """The best actions for a step's words inside each app it works in ("flag it" in an email flow -> the email
    mark action), kept by the context's trimming."""
    out = []
    for t in toolkits[:4]:
        try:
            hits = index.search(text, top, only={t})
        except TypeError:
            hits = []
        out += [e["id"] for e in hits if e["id"] in cat]
    return list(dict.fromkeys(out))


def container_of(flow, kind):
    """The loop / if a container step's piece ends with (its last top-level step of that action), or None."""
    want = _CONTAINER.get(kind)
    steps = flow.get("steps") if isinstance(flow, dict) else None
    for s in reversed(steps or []):
        if isinstance(s, dict) and s.get("action") == want:
            return s
    return None


def enclosing_loops(steps, i):
    out, p = [], steps[i]["parent"]
    while p is not None:
        if steps[p]["kind"] == "loop":
            out.insert(0, p)
        p = steps[p]["parent"]
    return out


_LEAD_EMAIL = re.compile(r"^(?:e-?mail|mail|send an? e-?mail to|send an? reminder to)\b", re.I)


# a step's words -> actions, where the step alone is too short for word search: (pattern, ids, pattern that must not
# match too or None)
STEP_HINTS = [
    (r"\b(?:ask|have|let|use|get)\s+(?:the\s+|an\s+)?(?:AI|GPT|LLM)\b|\b(?:AI|GPT|LLM)\s+(?:to\s+)?(?:decide|check|"
     r"classify|describe|judge|tell)", ["ooa.genai/classify", "ooa.genai/prompt"], r"screen ?shot|image|photo|picture"),
    (r"\b(?:ask|have|let|use|get)\s+(?:the\s+|an\s+)?AI\b.{0,60}\b(?:screen ?shots?|images?|photos?|pictures?|"
     r"shown)\b|\bdescribe\b.{0,40}\b(?:screen ?shots?|images?|photos?|shown)\b", ["ooa.genai/prompt_with_images"], None),
    (r"\b(?:the|an?|our|its)\s+(?:[\w']+\s+)?API\b|\bendpoint\b|\bREST\b", ["ooa.rest/request"], None),
    # "keep the ones whose outcome is Failed", "keep only quotes with status Sent": a filter, not a dedupe
    (r"^(?:keep|filter|select|pick)\b(?:\s+only)?\b.{0,40}?\b(?:with|whose|where|that (?:are|have|is|has))\b",
     ["ooa.datatable/filter"], r"\b(?:excel|sheet|workbook)\b"),
    (r"\b(?:collect|get|grab|extract|gather|list)\b.{0,30}\blinks?\b", ["ooa.browser/extract_list"], None),
    (r"\b(?:free|open|available)\s+(?:[\w-]+\s+){0,2}(?:slot|time)\b.{0,80}\b(?:outlook|microsoft|office 365|teams)\b|"
     r"\b(?:outlook|microsoft|teams)\b.{0,80}\b(?:free|available)\s+(?:[\w-]+\s+){0,2}slot", ["ooa.microsoft365/find_meeting_times"],
     None),
    (r"\bworking days?\b|\bbusiness days?\b|\bskipping (?:the )?(?:holidays|weekends)\b", ["ooa.datetime/add_working_days"],
     r"\bbetween\b"),
    (r"\b(?:more|older|longer) than \d+\s+(?:days?|weeks?|hours?|months?)\b|\b\d+\s+(?:days?|weeks?|hours?) ago\b|"
     r"\b(?:date|deadline) (?:has )?(?:passed|expired)\b|\bhas passed\b",
     ["ooa.datetime/now", "ooa.datetime/parse", "ooa.datetime/difference"], None),
]
_STEP_HINT_RX = [(re.compile(p, re.I), ids, re.compile(n, re.I) if n else None) for p, ids, n in STEP_HINTS]


_REPHRASE_EMAIL = re.compile(r"^(?:e-?mail|mail)\s+(.+?)\s+(how many|how|that|the|a|an|with|about|so|what|if|whether|"
                             r"saying)\b\s*(.*)$", re.I)


def rephrase(text):
    """The step in other words for a second try: "Email the HR ops team how many came in" -> "Send an email to the
    HR ops team about how many came in"; other steps as they were."""
    m = _REPHRASE_EMAIL.match(str(text or "").strip())
    if not m:
        return text
    who, word, rest = m.group(1), m.group(2).lower(), m.group(3)
    if word in ("so", "saying", "about"):
        return f"Send an email to {who} {word} {rest}".rstrip()
    return f"Send an email to {who} about {word} {rest}".rstrip()


def hint_ids(text):
    """Actions STEP_HINTS find in a step's words (what a guided sample starts with)."""
    t = str(text or "").strip()
    out = []
    for rx, ids, neg in _STEP_HINT_RX:
        if rx.search(t) and not (neg and neg.search(t)):
            out += ids
    return list(dict.fromkeys(out))


def step_verb_ids(text, strong=False):
    """Actions a step's own words point at (ooa_format.verb_hints, STEP_HINTS, and "email X ..." / "mail X ..." at
    its start: sending an email). strong: STEP_HINTS and the email start only (verb_hints is made for whole requests
    and too broad for one step: "drop the empty rows" is no file delete)."""
    t = str(text or "").strip()
    out = ([] if strong else list(F.verb_hints(t))) + hint_ids(t)
    if _LEAD_EMAIL.match(t):
        out.append("ooa.email/send")
    return list(dict.fromkeys(out))


def app_swaps(router, flow, named, hints, known, text):
    """A checked piece that uses an app nobody named where a named app has the same kind of action ("revoke the
    token" in a ServiceNow request answered with Salesforce's revoke; "wait until they're stopped" in an EC2 request
    answered with the file wait): the same piece with that app's action (one of the step's hints: same toolkit name
    or same first word of the action name), kept only when it passes the checker too. [] when nothing swaps."""
    if not named:
        return []
    by_name, by_verb = {}, {}
    for h in hints:
        t, _, name = str(h).partition("/")
        if t in named:
            by_name.setdefault(name, h)
            by_verb.setdefault(name.split("_")[0], h)
    out, compact = [], json.loads(F.compact(F.compact_flow(flow)))
    changed = False
    for s, _ in F.walk(compact.get("steps") or []):
        a = str(s.get("action", ""))
        t, _, name = a.partition("/")
        if not t.startswith("ooa.") or t in named:
            continue
        new = by_name.get(name) or by_verb.get(name.split("_")[0])
        if new and new.split("/")[0] != t and new in router.cat:
            # the arguments the new action takes are kept; a required one it lacks reads a variable of its name
            ins = router.cat[new].get("inputs") or []
            names = {x.get("name") for x in ins}
            args = {k: v for k, v in (s.get("args") or {}).items() if k in names}
            for x in ins:
                if x.get("required") and x.get("name") not in args:
                    args[x["name"]] = "$" + x["name"] + "$"
            s["action"], s["args"] = new, args
            outs = {o.get("name") for o in router.cat[new].get("outputs") or []}
            if isinstance(s.get("out"), dict):
                s["out"] = {k: v for k, v in s["out"].items() if k in outs}
            changed = True
    if changed:
        swapped, _why = router.verify(F.compact(compact), known, None, text)
        if swapped is not None:
            out.append(swapped)
    return out


def app_pick(router, text):
    """For a step that names its app itself ("Check on the statement in Snowflake", "Read the test points in Azure
    DevOps"): that app's best action for the step's words when the search over the whole catalog has it among its
    first 3 too (both searches agree), as [id]; else []."""
    named = F.named_apps(text)
    if len(named) != 1 or router.index is None:
        return []
    try:
        top = router.index.search(text, 1, only={named[0]})
    except TypeError:
        return []
    every = {e["id"] for e in router.index.search(text, 3)}
    if not top or top[0]["id"] not in every or top[0]["id"] not in router.cat:
        return []
    for verb in top[0]["id"].split("/")[-1].split("_"):
        if verb in SAID_VERBS and not re.search(r"\b" + verb, text, re.I):
            return []                     # "email ... from Gmail" is no Gmail delete, "find the instances" no reboot
    return [top[0]["id"]]


# an action of one of these verbs is never guided unless the step says the verb itself
SAID_VERBS = set("""delete remove reboot restart stop terminate cancel revoke purge drop destroy disable mark connect
disconnect end close archive move lock unlock start empty clear reset kill""".split())


def guided_prefix(router, text, kind, ids):
    """The given start of a guided sample's answer: one plan line per action the step's words point at (their plain
    words from the catalog), then for an if step its condition as an if, then "Flow:" (a thinking model gets the
    same as a thinking block). None when no action is in the catalog."""
    ids = [a for a in ids if a in router.cat]
    if not ids:
        return None
    steps = [{"action": a} for a in ids]
    if kind == "if":
        cond = re.sub(r"^(?:and\s+)?(?:then\s+)?(?:only\s+)?(?:if|when|in case|unless)\s+", "", text.split(",")[0],
                      flags=re.I).strip(" ,.")
        steps.append({"action": "core/if", "args": {"condition": cond}, "children": []})
    lines = F.plan_from_flow({"steps": steps}, router.cat)
    if kind == "if" and lines:
        lines[-1] = re.sub(r"^(\d+)\. .*\(core/if\)$", lambda m: f"{m.group(1)}. If {cond} (core/if)", lines[-1])
    think = getattr(router, "supports_think", None)
    return F.think_prefix(text, lines) if callable(think) and think() else F.answer_prefix(lines)


def fragment_score(flow, step, earlier, relevant, verb_ids=(), named=None):
    """How well a checked piece fits its step: an if step must end with an if (+3, else -3); actions among the
    step's own best catalog matches (+1 each, at most 2); every action an earlier piece already has with the same
    arguments (a repeated read or sign-in) -0.5; every action beyond 3 -0.5 (a piece that grew steps nobody asked
    for); a loop or an if the step did not ask for -1; an action its verb points at (step_verb_ids) +1.5; every
    action of an app neither the request nor the pieces so far name (named) -0.75."""
    acts = F.actions_of(flow)
    score = 0.0
    if step["kind"] == "if":
        score += 3.0 if container_of(flow, "if") is not None else -3.0
    score += min(2, len([a for a in acts if a in relevant]))
    seen = set()
    for b in earlier:
        for s, _ in F.walk(b.get("steps") or []):
            seen.add((s.get("action"), F.compact(s.get("args") or {})))
    for s, _ in F.walk(flow.get("steps") or []):
        if not str(s.get("action", "")).startswith("core/") and (s.get("action"), F.compact(s.get("args") or {})) in seen:
            score -= 0.5
    score -= 0.5 * max(0, len(acts) - 3)
    if verb_ids and any(a in verb_ids for a in acts):
        score += 1.5                       # what the step's verb says ("email the team" -> send an email)
    if named is not None:
        score -= 0.75 * len([a for a in acts if F.demoted(a, named)])    # an app nobody named
    if "core/loop" in acts and step["kind"] != "loop":
        score -= 1.0
    if "core/if" in acts and step["kind"] != "if" and not _ASKS_IF.search(step.get("text") or ""):
        score -= 1.0                       # "Only when the flag is true": an if the step did not ask for
    return score


_ASKS_IF = re.compile(r"\b(?:if|when|unless|only|otherwise|in case|whether)\b", re.I)


def unwrap(body, step, in_loop=False, item=None):
    """A do step's piece that is one if / loop the step did not ask for ("Only when the flag is true" around the real
    action, a loop again inside the loop it is in) -> the steps inside it. in_loop: the step is inside a loop already;
    a piece that gets its own list and loops over it again (file/list + core/loop + the real work) keeps only what
    its loop does."""
    if in_loop and step["kind"] == "do":
        tops = [k for k, x in enumerate(body) if isinstance(x, dict) and x.get("action") == "core/loop"
                and isinstance(x.get("children"), list) and x["children"]]
        if len(tops) == 1 and not body[tops[0]].get("else"):
            inner = body[tops[0]]["children"] + body[tops[0] + 1:]
            own = (body[tops[0]].get("args") or {}).get("as")
            if item and isinstance(own, str) and own and own != item:
                inner = F.rename_variable(inner, own, item)     # its own loop item is the enclosing loop's item
            return inner
    if (len(body) == 1 and isinstance(body[0], dict) and step["kind"] == "do"
            and body[0].get("action") in ("core/if", "core/loop") and isinstance(body[0].get("children"), list)
            and body[0]["children"] and not body[0].get("else")):
        if body[0]["action"] == "core/if" and _ASKS_IF.search(step.get("text") or ""):
            return body
        if body[0]["action"] == "core/loop" and step["parent"] is None and _EACH_INSIDE.search(step.get("text") or ""):
            return body
        return body[0]["children"]
    return body


def _drop_repeats(steps_list, done):
    """Top-level sign-in / connect / open steps an earlier piece already did (same action and arguments)."""
    out = []
    for s in steps_list:
        key = (s.get("action"), F.compact(s.get("args") or {}))
        if isinstance(s, dict) and _SETUP.search(str(s.get("action", ""))) and key in done:
            continue
        out.append(s)
    return out


def assemble(parts, steps):
    """The pieces joined into one compact flow: a piece goes into the body of the loop / if its step is inside (an
    if's then or else), after the steps there so far; a piece inside a loop that wrapped itself in the same loop again
    is unwrapped; a sign-in / connect step an earlier piece did is left out. A container step whose piece did not end
    with its container: the steps inside it go where the container would have gone. Returns {"steps": [...]}."""
    top, boxes, done = [], {}, set()
    for i, part in enumerate(parts):
        if part is None:
            continue
        body = json.loads(F.compact(F.compact_flow(part).get("steps") or []))
        s = steps[i]
        par = s["parent"]
        while par is not None and boxes.get(par) is None:
            par = steps[par]["parent"]
        target = top
        if par is not None:
            box = boxes[par]
            branch = s["branch"] if box.get("action") == "core/if" else "children"
            target = box.setdefault(branch, [])
        loops = enclosing_loops(steps, i)
        body = unwrap(body, s, in_loop=bool(loops), item=steps[loops[-1]].get("item_var") if loops else None)
        body = _drop_repeats(body, done)
        for x, _ in F.walk(body):
            done.add((x.get("action"), F.compact(x.get("args") or {})))
        target.extend(body)
        if s["kind"] in _CONTAINER:
            boxes[i] = container_of({"steps": body}, s["kind"])
            if s["kind"] == "if" and boxes[i] is not None and any(t["parent"] == i for t in steps[i + 1:]):
                # an if step is its condition ("if it's over 10,000"): what the model put inside its if is a guess;
                # the steps the request puts there come next
                boxes[i]["children"] = []
                boxes[i].pop("else", None)
    return {"steps": top}


def hoist_now(flow):
    """The current date and time read once, before the loops: a datetime/now step without arguments inside a loop
    moves to just before the outermost loop it is in (the training flows read it there)."""
    top = flow.get("steps") or []
    out = []
    for s in top:
        if isinstance(s, dict) and s.get("action") == "core/loop":
            moved = []

            def pull(body):
                keep = []
                for x in body:
                    if (isinstance(x, dict) and x.get("action") == "ooa.datetime/now" and not (x.get("args") or {})
                            and not moved):
                        moved.append(x)
                        continue
                    if isinstance(x, dict):
                        for k in ("children", "else"):
                            if isinstance(x.get(k), list):
                                x[k] = pull(x[k])
                    keep.append(x)
                return keep
            s["children"] = pull(s.get("children") or [])
            out.extend(moved)
        out.append(s)
    return {**flow, "steps": out}


def _made(flow):
    """Variables a piece makes (not inputs), in order: [variable dict]."""
    return [v for v in (flow or {}).get("variables") or [] if isinstance(v, dict) and v.get("direction") != "input"]


def loop_piece(collection, item, cat):
    """The code-built loop over a collection variable {"name", "type"}: a table first becomes records
    (ooa.datatable/to_records, as the training flows do), then core/loop over it with the item. A piece in the full
    form verify returns (variables + steps)."""
    name, steps, variables = collection["name"], [], []
    over = name
    if collection.get("type") == "Table" and "ooa.datatable/to_records" in cat:
        over = f"{item}_records" if f"{item}_records" != name else f"{item}_list"
        steps.append({"id": "s1", "action": "ooa.datatable/to_records", "args": {"table": f"${name}$"},
                      "out": {"records": over}})
        variables.append({"name": over, "type": "List", "direction": "local"})
    steps.append({"id": f"s{len(steps) + 1}", "action": "core/loop",
                  "args": {"iterator": "list", "over": f"${over}$", "as": item}, "children": []})
    variables.append({"name": item, "type": "Any", "direction": "local"})
    return {"variables": variables, "steps": steps}


def latest_collection(built, upto):
    """The newest Table / List variable the pieces before `upto` made, preferring the last step's ({"name", "type"})."""
    for b in reversed([b for b in built[:upto] if b is not None]):
        made = [v for v in _made(b) if v.get("type") in _COLLECTION]
        if made:
            # the last step's output when it is a collection, else the last collection declared
            last = (b.get("steps") or [{}])[-1]
            outs = [o for o in ((last.get("out") or {}).values() if isinstance(last.get("out"), dict) else [])
                    if isinstance(o, str)]
            for v in reversed(made):
                if v["name"] in outs:
                    return v
            return made[-1]
    return None


class _StepRun:
    """One step-by-step build (see build)."""

    def __init__(self, router, request, existing, tried, late, hard, steps, temperatures):
        self.r, self.request, self.tried, self.late, self.hard = router, request, tried, late, hard
        self.steps, self.temps = steps, temperatures
        self.base = [dict(v) for v in existing or [] if isinstance(v, dict)]
        self.built = [None] * len(steps)
        self.scores = [None] * len(steps)          # the score of each built piece (fragment_score + agreement)
        self.info = {"steps": [s["text"] for s in steps], "built": 0, "failed": [], "model_calls": 0}

    def known(self, i):
        known = list(self.base)
        for b in self.built[:i]:
            names = {v.get("name") for v in known}
            known += [v for v in (b or {}).get("variables") or [] if isinstance(v, dict) and v.get("name") not in names]
        for j in enclosing_loops(self.steps, i):
            item = self.steps[j].get("item_var")
            if item and item not in {v.get("name") for v in known}:
                known.append({"name": item, "type": "Any", "direction": "local"})
        return known

    def text(self, i):
        """The step's own request, a plain sentence as the training requests are: inside a loop, "it" / "them" / "each
        one" / "the item" become the loop's item ("the invoice PDF"), and a step that names neither gets "for the
        <item>" ("Read the total for the invoice PDF"); elsewhere a pronoun becomes the thing the step before worked on
        ("forward it" -> "forward the invoice"). LOOP_TEXT "with_current" gives "With the current <item>: ..." instead
        (the parts' wording)."""
        s = self.steps[i]
        loops = enclosing_loops(self.steps, i)
        t = s["text"].strip()
        if loops:
            lp = self.steps[loops[-1]]
            item = lp.get("item_var") or lp.get("item") or "item"
            if LOOP_TEXT == "with_current":
                t = resolve_pronouns(re.sub(r"\bthe item\b", f"the {item}", t), f"the {item}")
                return f"With the current {item}: {t[:1].lower() + t[1:]}"
            thing = lp.get("thing") or item.replace("_", " ")
            if thing.lower() in ("item", "items"):
                thing = item.replace("_", " ")
            head = thing.split()[-1].lower()
            t2 = re.sub(r"\b(?:each|every) (?:one|item|of them)\b|\bthe item\b", f"the {thing}", t, flags=re.I)
            t2 = re.sub(r"\b(?:each|every) " + re.escape(head) + r"\b", f"the {head}", t2, flags=re.I)
            t2 = resolve_pronouns(t2, f"the {thing}")
            if t2 == t and s["kind"] == "do" and head not in t.lower() and singular(head) not in t.lower():
                t2 = f"{t} for the {thing}"
            t = t2
        else:
            if _PRONOUN.search(t):
                t = resolve_pronouns(t, self.referent(i))
        return t[:1].upper() + t[1:]

    def referent(self, i):
        """What "it" / "them" in step i means: inside an if, the thing its condition talks about ("if the total is
        over ..."); else the thing the step before it at the same level works on (object_of), else the one before
        the if / loop it is in; never a person ("email the BI team ...") or a pronoun itself ("cancel it"). '' when
        unsure (the pronoun stays)."""
        s = self.steps[i]
        par = s["parent"]
        if par is not None and self.steps[par]["kind"] == "if":
            subj = subject_of(re.sub(r"^(?:and\s+)?(?:then\s+)?(?:only\s+)?(?:if|when|in case|unless)\s+", "",
                                     self.steps[par]["text"].strip(), flags=re.I))
            if subj and not _PRONOUN.search(subj):
                return subj
        for level in (par, self.steps[par]["parent"] if par is not None else "none"):
            if level == "none":
                break
            prev = [j for j in range(i) if self.steps[j]["kind"] == "do" and self.steps[j]["parent"] == level]
            if not prev:
                continue
            t = _LEAD.sub("", self.steps[prev[-1]]["text"].strip())
            if _TO_PERSON.match(t):
                return ""
            obj = object_of(t)
            return obj if obj and not _PRONOUN.search(obj) else ""
        return ""

    def plan_items(self):
        """Each loop's item variable, before anything is built (the steps inside it are asked at the same time as the
        others, so their request and known variables name it): the loop's noun ("for each rejected payment" ->
        payment); "for each one" -> the thing the step before it got ("the unread AP emails ..." -> email)."""
        names = {v.get("name") for v in self.base}
        for i, s in enumerate(self.steps):
            if s["kind"] != "loop":
                continue
            if not s.get("thing"):
                prev = [j for j in range(i) if self.steps[j]["kind"] == "do"]
                if prev:
                    s["thing"] = thing_of(object_of(self.steps[prev[-1]]["text"]))
            item = s.get("item") or "item"
            if item == "item" and s.get("thing"):
                item = item_name(s["thing"])
            while item in names:
                item += "_item"
            names.add(item)
            s["item_var"] = item

    def request_for(self, i):
        """(step request, kind it is built as) for step i: a loop over something no earlier step got asks for that
        collection first (its "source"), a loop over an earlier step's collection asks nothing (code builds it), and
        any other step asks its plain text (text)."""
        s = self.steps[i]
        if s["kind"] == "loop":
            if not s.get("refers") and s.get("source"):
                return s["source"][:1].upper() + s["source"][1:], "do"
            return None
        text = self.text(i)
        if s["kind"] == "if":
            # a condition alone is no request: it is asked with the first step inside it, as requests say it ("If
            # the message mentions a bug, open a Zendesk ticket"); that step's own piece replaces what the model
            # puts inside the if (assemble)
            kids = [j for j in range(i + 1, len(self.steps)) if self.steps[j]["parent"] == i
                    and self.steps[j]["branch"] == "children" and self.steps[j]["kind"] == "do"]
            if kids:
                t = self.text(kids[0])
                text = text.rstrip(" ,") + ", " + t[:1].lower() + t[1:]
        return text, s["kind"]

    def word_conditions(self):
        """An if step the model could not build, with steps of the request inside it: a core/if built by code whose
        condition is the request's own words ("the message mentions a bug, crash or error"), so the steps inside it
        keep their place; the notes ask for the expression. Returns [(step index, condition)]."""
        out = []
        for i, s in enumerate(self.steps):
            if s["kind"] != "if" or self.built[i] is not None:
                continue
            if not any(t["parent"] == i and b is not None for t, b in zip(self.steps, self.built)):
                continue
            cond = re.sub(r"^(?:and\s+)?(?:then\s+)?(?:only\s+)?(?:if|when|in case|unless)\s+", "", s["text"].strip(),
                          flags=re.I).strip(" ,.") or s["text"]
            self.built[i] = {"variables": [], "steps": [{"id": "s1", "action": "core/if", "args": {"condition": cond},
                                                         "children": []}]}
            out.append((i, cond))
        return out

    def weak(self, i):
        """A built do step whose piece has none of the actions its own words clearly point at (step_verb_ids strong:
        "email the team ..." without an email, "ask AI ..." without a genai action)."""
        if self.steps[i]["kind"] != "do" or self.built[i] is None:
            return False
        req = self.request_for(i)
        verbs = [a for a in step_verb_ids(req[0], strong=True) if a in self.r.cat] if req else []
        return bool(verbs) and not set(verbs) & set(F.actions_of(self.built[i]))

    def feeds_loop(self, i):
        """True when step i gets what a loop goes through: a loop's own source, or the last step before a loop over
        an earlier step's collection ("Read today's messages ..., for every message ...")."""
        s = self.steps[i]
        if s["kind"] == "loop":
            return True
        j = i + 1
        return (s["kind"] == "do" and j < len(self.steps) and self.steps[j]["kind"] == "loop"
                and bool(self.steps[j].get("refers")) and self.steps[j]["parent"] == s["parent"])

    def ask_many(self, asks, rnd, last):
        """One batched generation for several steps (asks: {i: (text, kind)}), each sampled at every temperature of
        self.temps; each answer checked (Router.verify) and the best checked piece of each step kept
        (fragment_score). last: {i: (previous answer, problems)} of steps asked again. Returns {i: piece or None}
        and records the problems of the steps that failed in last."""
        r = self.r
        items, owner, meta, guided_at = [], [], {}, {}
        for i, (text, kind) in asks.items():
            known = self.known(i)
            hints = step_hints(r.index, r.cat, text, step_toolkits(self.request, self.built[:i]))
            relevant = {e["id"] for e in r.index.search(text, 8)} | set(hints)
            prev, probs = last.get(i, (None, None))
            verbs = [a for a in step_verb_ids(text) if a in r.cat]
            need = list(dict.fromkeys(verbs + hints))
            prompt = r.prompt(text, known, prev, probs, required=need)
            free = prompt
            if SIMILAR_FIRST_ONLY and r.can_batch() and len(self.temps) > 1:
                try:
                    free = r.prompt(text, known, prev, probs, required=need, similar=False)
                except TypeError:                     # a router without similar=
                    free = prompt
            named = set(F.named_apps(self.request)) | set(F.named_apps(text))
            swap = step_hints(r.index, r.cat, text, [t for t in step_toolkits(self.request, self.built[:i])
                                                     if t in named], top=SWAP_TOP) if named else []
            meta[i] = (text, kind, known, relevant, verbs, named, self.feeds_loop(i), swap)
            temps = self.temps if r.can_batch() else self.temps[:1]
            for k, t in enumerate(temps):
                it = {"prompt": prompt if k == 0 else free, "attempt": rnd, "max_tokens": STEP_MAX_TOKENS}
                if r.can_batch():
                    it.update(temperature=t, seed=4000 + 97 * i + 13 * rnd + k)
                items.append(it)
                owner.append(i)
            strong = [a for a in hint_ids(text) if a in r.cat]
            bonus = GUIDED_BONUS
            if not strong and kind == "do":
                strong, bonus = app_pick(r, text), GUIDED_APP_BONUS
            guided = guided_prefix(r, text, kind, strong) if GUIDED_SAMPLE and strong and r.can_batch() else None
            if guided:
                guided_at[len(items)] = bonus
                items.append({"prompt": prompt, "attempt": rnd, "max_tokens": STEP_MAX_TOKENS, "prefix": guided,
                              "temperature": self.temps[0], "seed": 4000 + 97 * i + 13 * rnd + 11})
                owner.append(i)
        if not items:
            return {}
        outs = r.generate_many(items, self.hard)
        self.info["model_calls"] += len(items)
        self.info["batches"] = self.info.get("batches", 0) + 1
        best, fail, ok = {}, {}, {}
        for n, (i, (raw, _stop)) in enumerate(zip(owner, outs)):
            text, kind, known, relevant, verbs, named, feeds, hints = meta[i]
            ans = F.parse_answer(raw)
            if ans["kind"] != "flow":
                continue
            flow, why = r.verify(ans["flow_text"], known, ans["plan"], text, ans.get("facts"))
            if flow is None:
                fail.setdefault(i, (ans, why))
                continue
            step = dict(self.steps[i], kind=kind)
            acts = [a for a in F.actions_of(flow) if not str(a).startswith("core/")]
            if acts and max(acts.count(a) for a in acts) > MAX_SAME_ACTION:
                fail.setdefault(i, (ans, ["The same action is repeated; build the step once."]))
                continue                           # a runaway answer (the same action again and again)
            for k, cand in enumerate([flow] + app_swaps(r, flow, named, hints, known, text)):
                sc = fragment_score(cand, step, [b for b in self.built[:i] if b is not None], relevant, verbs, named)
                sc += 0.5 if k else 0.0            # the named app's own action
                sc += guided_at.get(n, 0.0)
                if feeds and any(v.get("type") in _COLLECTION for v in _made(cand)):
                    sc += 2.0                      # the loop after it needs a table or list to go through
                ok.setdefault(i, []).append((sc, cand))
        for i, cands in ok.items():
            # self-consistency: a piece whose actions other samples of the same step also chose gets AGREE each
            sigs = [tuple(F.actions_of(f)) for _, f in cands]
            for (sc, flow), sig in zip(cands, sigs):
                sc += AGREE * (sigs.count(sig) - 1)
                if i not in best or sc > best[i][0]:
                    best[i] = (sc, flow)
        out = {}
        for i in asks:
            text, kind, known = meta[i][:3]
            self.tried.append({"source": "local", "step": i + 1, "of": len(self.steps), "round": rnd,
                               "ok": i in best})
            if i in best and self.scores[i] is not None and self.built[i] is not None and best[i][0] <= self.scores[i]:
                out[i] = self.built[i]               # asked again (a weak piece): the earlier one was better
                continue
            if i in best:
                self.scores[i] = best[i][0]
            out[i] = best[i][1] if i in best else None
            if i not in best and i in fail:
                ans, why = fail[i]
                last[i] = (r._previous(ans["flow_text"], text, ans["plan"], known), why)
        return out

    def loop_piece(self, i):
        """A loop step's piece once the steps before it are built: the collection (the newest one an earlier step or
        its own source made) turned into records when it is a table, then core/loop with the item; None when there is
        no collection (the model then builds the loop: see run)."""
        s = self.steps[i]
        coll = latest_collection(self.built[:i + 1], i + 1)
        if coll is None:
            return None
        item = s.get("item_var") or "item"
        if item == coll["name"]:
            item = s["item_var"] = item + "_item"
        piece = loop_piece(coll, item, self.r.cat)
        src = self.built[i]
        if src is not None and container_of(src, "loop") is None:
            piece = {"variables": list(src.get("variables") or []) + piece["variables"],
                     "steps": list(src.get("steps") or []) + piece["steps"]}
        return piece

    def run(self):
        """Wave 1: every step at once (one batch: the source of each loop over something new, each other step's
        plain request, the steps inside a loop knowing its item); then the loops are built by code over the newest
        collection before them; then up to STEP_ROUNDS - 1 more waves ask the failed steps again (one batch), each
        with its previous answer and problems and the variables the steps before it made; a loop without a
        collection is then asked of the model as a step. The pieces are joined (assemble) with variables linked."""
        self.plan_items()
        last = {}
        for rnd in range(STEP_ROUNDS):
            if self.late():
                return None, [OUT_OF_TIME], self.info
            asks = {}
            for i, s in enumerate(self.steps):
                if s["kind"] == "loop":
                    if self.built[i] is not None and container_of(self.built[i], "loop") is not None:
                        continue
                    req = self.request_for(i)
                    if rnd == 0 or (req is not None and self.built[i] is None):
                        if req is not None:
                            asks[i] = req                  # the loop's collection first
                    else:
                        asks[i] = (s["text"][:1].upper() + s["text"][1:], "loop")     # the model builds the loop
                elif self.built[i] is None:
                    asks[i] = self.request_for(i)
                elif rnd > 0 and self.weak(i):
                    # built, but no sample did what its verb says ("email the team ..." without an email): asked
                    # again in other words; the new piece replaces it only when it scores higher
                    text, kind = self.request_for(i)
                    asks[i] = (rephrase(text), kind)
            try:
                got = self.ask_many(asks, rnd, last)
            except Exception as ex:                       # a backend error ends the step path, never the request
                return None, [f"local model error: {ex}"], self.info
            for i, piece in got.items():
                if piece is None:
                    continue
                if self.steps[i]["kind"] == "loop" and asks[i][1] == "loop":
                    box = container_of(piece, "loop")
                    if box is None:
                        continue
                    self.steps[i]["item_var"] = (box.get("args") or {}).get("as") or self.steps[i].get("item_var")
                self.built[i] = piece
            for i, s in enumerate(self.steps):
                if s["kind"] == "loop" and (self.built[i] is None or container_of(self.built[i], "loop") is None):
                    lp = self.loop_piece(i)
                    if lp is not None:
                        self.built[i] = lp
            if all(b is not None and (st["kind"] != "loop" or container_of(b, "loop") is not None)
                   for b, st in zip(self.built, self.steps)) and not any(self.weak(i) for i in range(len(self.steps))):
                break
        worded = self.word_conditions()
        n = len(self.steps)
        self.info["failed"] = [s["text"] for s, b in zip(self.steps, self.built) if b is None]
        self.info["built"] = n - len(self.info["failed"])
        if not self.info["built"] or len(self.info["failed"]) > MAX_FAILED_SHARE * n:
            return None, [f"{len(self.info['failed'])} of {n} steps could not be built: "
                          + "; ".join(self.info["failed"][:3])], self.info
        have = [b for b in self.built if b is not None]
        linked, links = F.link_part_variables(have, self.base)
        it = iter(linked)
        parts = [next(it) if b is not None else None for b in self.built]
        joined = hoist_now(assemble(parts, self.steps))
        flow, notes = self.r.verify(F.compact(joined), self.base, None, self.request)
        self.tried.append({"source": "local", "steps_joined": self.info["built"], "ok": flow is not None})
        if flow is None:
            return None, ["The flow built step by step did not pass the checker: " + "; ".join(notes[:2])], self.info
        return flow, list(notes) + [f"built step by step: {self.info['built']} of {n} steps"] + [
            f"step {k + 1}: its condition is the request's words ('{c}'): write it as an expression" for k, c in worded] + [
            f"step {k}: '{a}' is '{b}' of an earlier step" for k, a, b in links], self.info


OUT_OF_TIME = "The time for this request ran out, so no further attempt was made."


def build(router, request, existing, tried, late, hard, steps=None, temperatures=STEP_TEMPERATURES):
    """Build a flow step by step (see the module doc). router: serving's Router (prompt, verify, generate_many,
    index, cat). Returns (flow, notes, info) when the joined flow passes the checker, else (None, problems, info).
    info: {"steps": [the step texts], "built": n, "failed": [texts], "model_calls", "seconds"}."""
    t0 = time.time()
    steps = [dict(s) for s in (steps if steps is not None else split_steps(request))]
    if len(steps) < 2:
        return None, ["the request has fewer than 2 steps"], {"steps": [s["text"] for s in steps], "built": 0,
                                                              "failed": [], "model_calls": 0}
    flow, notes, info = _StepRun(router, request, existing, tried, late, hard, steps, temperatures).run()
    info["seconds"] = round(time.time() - t0, 2)
    return flow, notes, info


# -- when to use it, and which flow to keep ------------------------------------------------------------------------
WANT_STEPS = 3                 # a request of this many steps (loops aside) or more ...
WANT_WORDS = 25                # ... and this many words, or that names 2+ apps, is built step by step too


def wants(request):
    """Auto / smart / ultra run the step builder for a long or multi-app request: WANT_STEPS steps or more (loops not
    counted) with WANT_WORDS words or more or a loop / if among them, or 2+ named apps with 2+ steps. A story ("First I
    open ..., then I ...") in fewer than 2 named apps is left to the single answer."""
    steps = split_steps(request)
    n = len([s for s in steps if s["kind"] != "loop"])
    words = len(str(request or "").split())
    control = any(s["kind"] in ("loop", "if") for s in steps)
    if is_story(request) and len(F.named_apps(request)) < 2:
        return False
    return (n >= WANT_STEPS and (words >= WANT_WORDS or control)) or (n >= 2 and len(F.named_apps(request)) >= 2)


def coverage(router, request, flow):
    """How much of the request a flow covers: (request clauses no step covers, ooa_format.uncovered_clauses), lower
    is better."""
    try:
        return len(F.uncovered_clauses(request, flow, router.index))
    except Exception:
        return 99


PREFER_TIE_STEPS = 4            # a request of this many steps or more (loops aside) is a long one


def prefer(router, request, stepped, single, steps=None):
    """Keep the step-built flow over the single answer's? Yes when the single answer did not verify, or the step-built
    one leaves no more request clauses uncovered (coverage). A tie goes to the step-built flow: it was built from the
    request's own steps, while the single answer of a long request tends to follow a similar example (measured on
    plain_requests_3 with the v4 model: ties to the step-built flow 93 gold actions of 160, ties to the single
    answer for requests of fewer than 4 steps 89, the single answer alone 78)."""
    if single is None:
        return True
    return coverage(router, request, stepped) <= coverage(router, request, single)


def long_request(request, steps=None):
    """PREFER_TIE_STEPS steps or more, loops aside."""
    steps = split_steps(request) if steps is None else steps
    return len([s for s in steps if s["kind"] != "loop"]) >= PREFER_TIE_STEPS
