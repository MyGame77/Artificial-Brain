import re

from knowledge import KnowledgeEngine
from azure_knowledge import AzureKnowledgeEngine
from ollama_engine import OllamaEngine, DEFAULT_MODEL


MAX_KNOWLEDGE_RESULTS = 8

FACT_VERBS = {
    "is", "are", "was", "were", "means", "contains", "contain", "uses", "use",
    "converts", "convert", "produces", "produce", "creates", "create", "created",
    "stores", "store", "generates", "generate", "measures", "measure", "requires",
    "require", "allows", "allow", "causes", "cause", "has", "have", "forms",
    "form", "consists", "consist", "depends", "depend", "flows", "flow",
    "provides", "provide", "connects", "connect", "powers", "power", "runs",
    "run", "built", "builds", "build", "founded", "founds", "invented",
    "invents", "discovered", "discovers", "designed", "designs", "developed",
    "develops", "includes", "include", "supports", "support", "controls",
    "control", "owns", "own", "orbits", "orbit", "freezes", "freeze",
    "boils", "boil", "melts", "melt", "evaporates", "evaporate", "rotates",
    "revolves", "travels", "travel", "attracts", "attract", "repels", "repel",
    "increases", "increase", "decreases", "decrease", "moves", "move",
    "forms", "form", "occurs", "occur", "exists", "exist", "becomes", "become",
}
PERSONAL_OR_CHAT_STARTS = (
    "i ", "i'm ", "im ", "my ", "mine ", "we ", "we're ", "our ",
    "you ", "your ", "he ", "she ", "they ", "someone ", "my friend ",
    "wow ", "oh ", "ah ", "aha ", "haha", "lol", "lmao", "nice ",
    "cool ", "great ", "awesome ", "bro ", "thanks", "thank you",
    "hello", "hi ", "hey ", "yes ", "yeah ", "yep", "no ", "nope",
    "okay", "ok ", "alright", "hmm", "umm", "uh ",
)
SENSITIVE_PATTERNS = (
    r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b",
    r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)",
    r"\b(password|passcode|pin|address|phone number|date of birth)\b",
)


def infer_topic(sentence, verb):
    parts = re.split(rf"\b{re.escape(verb)}\b", sentence, maxsplit=1, flags=re.I)
    if len(parts) > 1 and verb in {
        "created", "create", "creates", "built", "build", "builds",
        "founded", "founds", "invented", "invents", "designed", "designs",
        "developed", "develops",
    }:
        source = parts[1]
    else:
        source = parts[0]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", source)
    words = [word for word in words if word.lower() not in {"a", "an", "the"}]
    return " ".join(word.capitalize() for word in words[-3:]) or "General Knowledge"


def auto_learn_text(prompt, knowledge):
    """Learn suitable facts from ordinary messages, notes, and pasted text."""
    text = prompt.strip()
    if not text or len(text) > 12000:
        return []

    # Natural-language requests to remember a fact are accepted without slash commands.
    text = re.sub(r"(?im)^\s*(?:please\s+)?(?:remember|learn|note)\s+that\s+", "", text)
    candidates = re.split(r"(?<=[!?])\s+|(?<!\d)\.(?=\s+[A-Z])\s+|\n+|\s*\*\s+|\s*[-•]\s+", text)
    learned = []
    for raw in candidates:
        sentence = " ".join(raw.strip(" \t-*•") .split())
        if len(sentence) < 20 or len(sentence) > 500 or "?" in sentence:
            continue
        lowered = sentence.casefold().strip(" .!\t")
        if lowered.startswith(PERSONAL_OR_CHAT_STARTS):
            continue
        if lowered.startswith((
            "this is just ", "this is awesome", "this is great", "this is cool",
            "this is nice", "this is funny", "this is random",
        )):
            continue
        if lowered.startswith((
            "what ", "who ", "whom ", "whose ", "where ", "when ", "why ", "how ",
            "which ", "is ", "are ", "am ", "was ", "were ", "do ", "does ", "did ",
            "can ", "could ", "would ", "will ", "should ", "have ", "has ", "had ",
            "tell me ", "explain ", "describe ", "show me ", "give me ",
            "i think ", "maybe ", "perhaps ", "i guess ", "i heard ",
        )):
            continue
        if any(re.search(pattern, lowered) for pattern in SENSITIVE_PATTERNS):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        if len(words) < 5:
            continue
        verb = next((v for v in sorted(FACT_VERBS, key=len, reverse=True)
                     if re.search(rf"\b{re.escape(v)}\b", lowered)), None)
        if not verb:
            continue
        try:
            result = knowledge.add_fact(
                infer_topic(sentence, verb), sentence,
                source="user-provided-auto-learn", verified=False,
            )
            if result[1]:
                learned.append(result[0])
        except ValueError:
            continue
    return learned


def make_memory_context(facts):
    if not facts:
        return ""
    return "\n".join(
        f"[{fact.get('kind', 'local fact').title()} #{fact['id']} | "
        f"{fact['topic']} | source: {fact.get('source', 'local')} | "
        f"verified: {fact.get('verified', 'not specified')}] {fact['content']}"
        for fact in facts
    )


def split_topic_fact(value):
    if "|" not in value:
        raise ValueError("Use: /learn Topic | fact text")
    topic, information = value.split("|", 1)
    return topic.strip(), information.strip()


def handle_command(prompt, brain, local_knowledge, knowledge):
    command, _, argument = prompt.partition(" ")
    command = command.lower()
    argument = argument.strip()

    if command == "/help":
        print("\nKnowledge commands:")
        print("  /knowledge [search words]  Browse or search saved facts and memories")
        print("  /forget F3 or M2            Delete a Knowledge fact or memory")
        print("  /forget-local ID            Delete an old local memory")
        print("  /filter list                Show blocked terms")
        print("  /filter add TERM            Block a term from learning/retrieval")
        print("  /filter remove TERM         Remove a blocked term")
        print("  /rule list                  Show answer rules")
        print("  /rule add TEXT              Add an answer rule")
        print("  /rule remove TEXT           Remove an answer rule")
        print("\nOther commands: /model, /model NAME, /clear, /help, /exit")
        print("Knowledge learns suitable facts automatically from normal messages and pasted notes.")
        return True

    if command == "/knowledge":
        records = knowledge.list_records()
        if argument:
            terms = KnowledgeEngine.tokenize(argument)
            records = [r for r in records if terms & (
                KnowledgeEngine.tokenize(r[1]) | KnowledgeEngine.tokenize(r[2])
            )]
        print(f"\nAzure Knowledge: {knowledge.count_facts()} facts, "
              f"{knowledge.count_memories()} memories")
        if not records:
            print("No matching saved records.")
        for record_id, topic, content in records:
            print(f"{record_id} | {topic}: {content}")
        local = local_knowledge.list_facts()
        print(f"\nLegacy local memory: {len(local)} facts")
        for item in local:
            if not argument or KnowledgeEngine.tokenize(argument) & (
                KnowledgeEngine.tokenize(item["topic"]) |
                KnowledgeEngine.tokenize(item["content"])
            ):
                print(f"L{item['id']} | {item['topic']}: {item['content']}")
        return True

    if command == "/forget":
        if knowledge.delete_record(argument):
            print(f"Deleted Knowledge record {argument.upper()}.")
        else:
            print("Use /forget F3 or /forget M2. Record not found.")
        return True

    if command == "/forget-local":
        if argument.isdigit() and local_knowledge.delete_fact(int(argument)):
            print(f"Deleted local memory #{argument}.")
        else:
            print("Use /forget-local ID. Local record not found.")
        return True

    if command == "/filter":
        action, _, value = argument.partition(" ")
        action, value = action.lower(), value.strip()
        try:
            if action == "list":
                terms = knowledge.list_filters()
                print("Blocked terms: " + (", ".join(terms) if terms else "(none)"))
            elif action == "add":
                print("Filter added." if knowledge.add_filter(value)
                      else "That filter already exists.")
            elif action == "remove":
                print("Filter removed." if knowledge.remove_filter(value)
                      else "Filter not found.")
            else:
                print("Use /filter list, /filter add TERM, or /filter remove TERM.")
        except (ValueError, RuntimeError) as error:
            print(f"Filter error: {error}")
        return True

    if command == "/rule":
        action, _, value = argument.partition(" ")
        action, value = action.lower(), value.strip()
        try:
            if action == "list":
                rules = knowledge.list_rules()
                for index, rule in enumerate(rules, 1):
                    print(f"{index}. {rule}")
                if not rules:
                    print("No answer rules.")
            elif action == "add":
                print("Rule added." if knowledge.add_rule(value)
                      else "That rule already exists.")
            elif action == "remove":
                if knowledge.remove_rule(value):
                    print("Rule removed.")
                else:
                    print("Rule not found.")
            else:
                print("Use /rule list, /rule add TEXT, or /rule remove TEXT.")
        except (ValueError, RuntimeError) as error:
            print(f"Rule error: {error}")
        return True

    if command == "/model":
        try:
            if argument:
                brain.set_model(argument)
                print(f"Active model changed to {brain.model}. Conversation context cleared.")
            else:
                print(f"Active model: {brain.model}")
                print("Installed models: " + ", ".join(brain.list_models()))
        except (RuntimeError, ValueError) as error:
            print(f"Model error: {error}")
        return True

    if command == "/clear":
        brain.reset_history()
        print("Conversation context cleared. Long-term Knowledge was kept.")
        return True

    return False


def main():
    print("=" * 56)
    print("                    ARTIFICIAL BRAIN")
    print("               Local AI + Knowledge system")
    print("=" * 56)

    local_knowledge = KnowledgeEngine()
    knowledge = AzureKnowledgeEngine()
    brain = OllamaEngine(model=DEFAULT_MODEL)

    try:
        fact_count = knowledge.count_facts()
        memory_count = knowledge.count_memories()
        rules = knowledge.rules_context()
        filter_count = len(knowledge.list_filters())
    except RuntimeError as error:
        print(f"Knowledge startup error: {error}")
        return

    try:
        installed = brain.list_models()
    except RuntimeError as error:
        print(f"Startup error: {error}")
        return
    if brain.model not in installed:
        if not installed:
            print("No local Ollama models are installed.")
            return
        brain.model = installed[0]

    print("Local model:", brain.model)
    print("Azure Knowledge:", fact_count, "facts,", memory_count, "memories (editable copy)")
    print("Knowledge rules:", len(knowledge.list_rules()), "| active filters:", filter_count)
    print("Legacy local memory:", len(local_knowledge.list_facts()), "facts")
    print("Learning: automatic from normal text and pasted notes")
    print("Type /help for commands.")
    print("-" * 56)

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if not prompt:
                continue
            if prompt.lower() in {"/exit", "exit", "quit"}:
                print("Artificial Brain: Goodbye.")
                break
            if prompt.startswith("/"):
                if not handle_command(prompt, brain, local_knowledge, knowledge):
                    print("Unknown command. Type /help.")
                continue

            blocked = knowledge.blocked_match(prompt)
            if blocked:
                print(f"Artificial Brain: That message matches the active Knowledge filter '{blocked}'.")
                continue

            learned = auto_learn_text(prompt, knowledge)
            if learned:
                print(f"[Knowledge: learned {len(learned)} new fact{'s' if len(learned) != 1 else ''} from your text]")

            azure_facts = knowledge.search(prompt, limit=MAX_KNOWLEDGE_RESULTS)
            local_facts = local_knowledge.search(
                prompt, limit=max(0, MAX_KNOWLEDGE_RESULTS - len(azure_facts))
            )
            context = make_memory_context(azure_facts + local_facts)
            # Reload rules so additions/removals take effect immediately.
            rules = knowledge.rules_context()
            if rules:
                context = (
                    "Knowledge answer rules (follow when relevant):\n"
                    + rules + "\n\n" + context
                )
            print("\nArtificial Brain: ", end="", flush=True)
            brain.respond(
                prompt,
                knowledge_context=context,
                on_token=lambda token: print(token, end="", flush=True),
            )
            print()

        except KeyboardInterrupt:
            print("\nArtificial Brain: Goodbye.")
            break
        except (RuntimeError, ValueError) as error:
            print(f"\nArtificial Brain error: {error}")


if __name__ == "__main__":
    main()