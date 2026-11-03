import json
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "knowledge" / "knowledge.json"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "who", "whom", "whose", "where", "when", "why", "how", "does", "do", "did",
    "tell", "me", "about", "this", "that", "your", "you", "it", "of",
    "to", "and", "in", "on", "for", "with", "please", "can", "could",
    "would", "should", "explain", "describe", "show", "give", "say",
}

CONCEPT_ALIASES = {
    "electricity": "electrical",
    "electric": "electrical",
    "transformer": "architecture",
    "transformers": "architecture",
    "architectural": "architecture",
    "design": "architecture",
    "structure": "architecture",
    "database": "knowledge",
    "databases": "knowledge",
    "fact": "knowledge",
    "facts": "knowledge",
    "information": "knowledge",
    "memory": "knowledge",
    "memories": "knowledge",
    "stores": "store",
    "stored": "store",
    "saving": "store",
    "save": "store",
    "saved": "store",
    "keeps": "store",
    "keep": "store",
    "developed": "develop",
    "developing": "develop",
    "development": "develop",
    "created": "create",
    "creates": "create",
    "creating": "create",
    "built": "build",
    "building": "build",
    "founded": "found",
    "founder": "found",
    "invented": "invent",
    "inventing": "invent",
    "discovered": "discover",
    "designed": "design",
    "named": "name",
    "called": "call",
    "creator": "create",
}


class KnowledgeEngine:
    def __init__(self):
        DATABASE.parent.mkdir(parents=True, exist_ok=True)
        self.facts = []
        self.load()

    def load(self):
        if not DATABASE.exists():
            self.save()
            return

        try:
            data = json.loads(DATABASE.read_text(encoding="utf-8"))

            if not isinstance(data, list):
                raise ValueError("Knowledge database must be a list.")

            seen_ids = set()
            for fact in data:
                if not isinstance(fact, dict):
                    raise ValueError("Every knowledge record must be an object.")
                if not {"id", "topic", "content"}.issubset(fact):
                    raise ValueError("A knowledge record is missing required fields.")
                if not isinstance(fact["id"], int) or fact["id"] <= 0:
                    raise ValueError("Knowledge fact IDs must be positive integers.")
                if fact["id"] in seen_ids:
                    raise ValueError("Knowledge fact IDs must be unique.")
                if not isinstance(fact["topic"], str) or not isinstance(fact["content"], str):
                    raise ValueError("Knowledge topics and contents must be text.")
                if not fact["topic"].strip() or not fact["content"].strip():
                    raise ValueError("Knowledge topics and contents cannot be empty.")
                seen_ids.add(fact["id"])

            self.facts = data

        except (json.JSONDecodeError, OSError) as error:
            raise RuntimeError(
                f"Could not load knowledge database: {error}"
            ) from error

    def save(self):
        temporary = DATABASE.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(self.facts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(DATABASE)

    @staticmethod
    def tokenize(text):
        raw_tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        concepts = set()

        for token in raw_tokens:
            if token in STOPWORDS:
                continue

            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
                token = token[:-1]

            token = CONCEPT_ALIASES.get(token, token)
            concepts.add(token)

        return concepts

    def add_fact(self, topic, content):
        if not isinstance(topic, str) or not isinstance(content, str):
            raise ValueError("Topic and fact must be text.")

        topic = topic.strip()
        content = content.strip()

        if not topic or not content:
            raise ValueError("Topic and fact cannot be empty.")
        if len(topic) > 100:
            raise ValueError("Topic must be 100 characters or fewer.")
        if len(content) > 2000:
            raise ValueError("A fact must be 2000 characters or fewer.")

        normalized_topic = topic.casefold()
        normalized_content = re.sub(r"\W+", " ", content.casefold()).strip()

        for existing in self.facts:
            old_content = re.sub(
                r"\W+", " ", existing["content"].strip().casefold()
            ).strip()
            if old_content == normalized_content:
                return existing, False

        next_id = max(
            (fact["id"] for fact in self.facts),
            default=0,
        ) + 1

        fact = {
            "id": next_id,
            "topic": topic,
            "content": content,
        }

        self.facts.append(fact)
        self.save()
        return fact, True

    def search(self, query, limit=5, min_score=0.35):
        if limit <= 0:
            return []

        query_concepts = self.tokenize(query)
        if not query_concepts:
            return []

        ranked = []

        for fact in self.facts:
            topic_concepts = self.tokenize(fact["topic"])
            content_concepts = self.tokenize(fact["content"])
            fact_concepts = topic_concepts | content_concepts

            matched = query_concepts & fact_concepts
            if not matched:
                continue

            coverage = len(matched) / len(query_concepts)
            topic_bonus = 0.20 if query_concepts & topic_concepts else 0.0
            phrase_bonus = 0.10 if (
                fact["topic"].strip().lower() in query.strip().lower()
                or query.strip().lower() in fact["topic"].strip().lower()
            ) else 0.0

            score = coverage + topic_bonus + phrase_bonus
            if score >= min_score:
                ranked.append((score, fact["id"], fact))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [fact for _, _, fact in ranked[:limit]]

    def list_facts(self):
        return list(self.facts)

    def delete_fact(self, fact_id):
        original_count = len(self.facts)
        self.facts = [
            fact for fact in self.facts
            if fact["id"] != fact_id
        ]

        if len(self.facts) == original_count:
            return False

        self.save()
        return True


def main():
    engine = KnowledgeEngine()

    print("=" * 40)
    print("       ARTIFICIAL BRAIN")
    print("       Knowledge Manager")
    print("=" * 40)
    print("Commands:")
    print("add     - Add a fact")
    print("search  - Search facts")
    print("list    - List all facts")
    print("delete  - Delete a fact by ID")
    print("exit    - Quit")
    print("-" * 40)

    while True:
        try:
            command = input("Knowledge> ").strip().lower()

            if command == "exit":
                print("Knowledge manager closed.")
                break

            if command == "add":
                topic = input("Topic: ").strip()
                content = input("Fact: ").strip()
                fact, created = engine.add_fact(topic, content)
                if created:
                    print(f"Saved fact #{fact['id']}.")
                else:
                    print(f"That fact is already stored as #{fact['id']}.")

            elif command == "search":
                query = input("Search: ").strip()
                results = engine.search(query)

                if not results:
                    print("No relevant facts found.")

                for fact in results:
                    print(f"\nID: {fact['id']}")
                    print(f"Topic: {fact['topic']}")
                    print(f"Fact: {fact['content']}")

            elif command == "list":
                facts = engine.list_facts()
                if not facts:
                    print("Knowledge database is empty.")

                for fact in facts:
                    print(f"\n#{fact['id']} | {fact['topic']}")
                    print(fact["content"])

            elif command == "delete":
                value = input("Fact ID: ").strip()
                if not value.isdigit():
                    print("Please enter a valid numeric ID.")
                elif engine.delete_fact(int(value)):
                    print("Fact deleted.")
                else:
                    print("Fact ID not found.")

            elif command:
                print("Unknown command.")

        except (ValueError, RuntimeError) as error:
            print(f"Error: {error}")
        except KeyboardInterrupt:
            print("\nKnowledge manager closed.")
            break


if __name__ == "__main__":
    main()