import spacy
import re
import json
import csv
import numpy as np

# Load NER model
nlp = spacy.load("en_core_web_sm")


class StatisticalHelper:
    """
    Helper to convert ValidationLayer heuristic scores into
    calibrated probabilistic reliability estimates using
    the core ideas of temperature scaling + batch prior.
    """
    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def temperature_scale(self, logits):
        """Simple sigmoid temperature scaling"""
        return 1 / (1 + np.exp(-logits / self.temperature))

    def calibrate_scores(self, audits):
        """
        Convert total_score to calibrated probability [0,1]
        audits: list of audit dicts from ValidationLayer.process_top_k
        """
        logits = np.array([a['total_score'] for a in audits])
        probs = self.temperature_scale(logits)
        return probs

    def adjust_by_batch_prior(self, probs):
        """
        Empirical Bayes style batch adjustment.
        Assume batch prior = mean probability of top-k docs
        Adjust individual probabilities towards batch mean
        """
        prior = np.mean(probs)
        adjusted = 0.7 * probs + 0.3 * prior  # 0.7 individual weight, 0.3 batch
        return np.clip(adjusted, 0.0, 1.0)

    def apply_fdr_threshold(self, probs, accept_thresh=0.65, abstain_thresh=0.45):
        """
        Apply probabilistic thresholds to decide
        accept / abstain / reject
        """
        decisions = []
        for p in probs:
            if p >= accept_thresh:
                decisions.append("accept")
            elif p >= abstain_thresh:
                decisions.append("abstain")
            else:
                decisions.append("reject")
        return decisions


class ValidationLayer:
    def __init__(self, save_json_path="audit_results.json", save_csv_path="audit_results.csv"):
        self.save_json_path = save_json_path
        self.save_csv_path = save_csv_path
        self.stat_helper = StatisticalHelper(temperature=0.9)  # helper for probabilistic decisions

    # -----------------------------
    # Stage B: Heuristic Checks
    # -----------------------------

    def check_entity(self, doc, query):
        query_entities = [ent.text.lower() for ent in nlp(query).ents]
        doc_entities = [ent.text.lower() for ent in nlp(doc).ents]

        if not query_entities:
            query_keywords = set(re.findall(r'\w+', query.lower()))
            doc_keywords = set(re.findall(r'\w+', doc.lower()))
            overlap = query_keywords.intersection(doc_keywords)
            return "yes" if len(overlap) / max(len(query_keywords), 1) > 0.3 else "no"

        overlap = set(query_entities).intersection(set(doc_entities))
        return "yes" if overlap else "no"

    def check_completeness(self, doc, query):
        query_doc = nlp(query)
        expected_keywords = [tok.lemma_.lower() for tok in query_doc if tok.pos_ in ("NOUN", "PROPN", "VERB")]
        doc_text = doc.lower()
        count = sum(1 for kw in expected_keywords if kw in doc_text)
        coverage = count / max(len(expected_keywords), 1)
        return "sufficient" if coverage > 0.6 else "insufficient"

    def evaluate_confidence(self, doc):
        high_confidence_words = ["definitely", "without a doubt", "for sure", "ultimately", "confirmed"]
        count = sum(doc.lower().count(word) for word in high_confidence_words)
        if count == 0:
            return "low"
        elif count <= 2:
            return "medium"
        else:
            return "high"

    def check_conflict(self, docs):
        contradictions = False
        pattern = re.compile(r'(\w+)\s+(marries|is|becomes)\s+(\w+)', re.IGNORECASE)
        entity_map = {}
        for doc in docs:
            matches = pattern.findall(doc)
            for e1, verb, e2 in matches:
                key = (e1.lower(), verb.lower())
                if key in entity_map and entity_map[key] != e2.lower():
                    contradictions = True
                else:
                    entity_map[key] = e2.lower()
        return "yes" if contradictions else "no"

    # -----------------------------
    # Stage B: Audit + Stage C: Decision Gate
    # -----------------------------

    def audit_document(self, doc, query, all_docs=None):
        entity_match = self.check_entity(doc, query)
        completeness = self.check_completeness(doc, query)
        internal_conflict = self.check_conflict(all_docs) if all_docs else "no"
        confidence_score = self.evaluate_confidence(doc)

        # Convert qualitative confidence to numeric
        confidence_map = {"low": 0.3, "medium": 0.6, "high": 0.9}
        completeness_map = {"insufficient": 0.3, "sufficient": 0.7}
        entity_map = {"no": 0.3, "yes": 0.8}
        conflict_map = {"no": 0.8, "yes": 0.2}

        total_score = (
            confidence_map[confidence_score] +
            completeness_map[completeness] +
            entity_map[entity_match] +
            conflict_map[internal_conflict]
        ) / 4  # average score as pseudo-logit

        return {
            "document": doc,
            "entity_match": entity_match,
            "completeness": completeness,
            "internal_conflict": internal_conflict,
            "confidence_score": confidence_score,
            "total_score": total_score
        }

    def decision_gate(self, audit):
        """Original heuristic fallback if needed"""
        if audit["entity_match"] == "no":
            return "reject"
        elif audit["internal_conflict"] == "yes":
            return "abstain"
        elif audit["completeness"] == "insufficient":
            return "abstain"
        else:
            return "accept"

    # -----------------------------
    # Stage C/D: Process Top-K Docs
    # -----------------------------

    def process_top_k(self, docs, query):
        results = []
        for doc in docs:
            audit = self.audit_document(doc, query, all_docs=docs)
            results.append(audit)

        # -----------------------------
        # Apply Statistical Helper for calibrated decisions
        # -----------------------------
        probs = self.stat_helper.calibrate_scores(results)
        probs_adj = self.stat_helper.adjust_by_batch_prior(probs)
        stat_decisions = self.stat_helper.apply_fdr_threshold(probs_adj)

        # Merge with original heuristic decision
        for i, audit in enumerate(results):
            original_decision = self.decision_gate(audit)
            if stat_decisions[i] == "accept":
                audit["decision"] = "accept"
            elif stat_decisions[i] == "reject":
                audit["decision"] = "reject"
            else:
                audit["decision"] = original_decision  # keep heuristic decision

        # Save JSON and CSV for audit + evaluation
        self.save_results(results, query)

        # Return only accepted documents for generator
        filtered_docs = [r["document"] for r in results if r["decision"] == "accept"]
        return filtered_docs, results

    # -----------------------------
    # File Saving Utility
    # -----------------------------

    def save_results(self, results, query):
        """Save audit results for further analysis."""
        with open(self.save_json_path, "a") as f_json:
            json.dump({"query": query, "results": results}, f_json, indent=2)
            f_json.write("\n")

        with open(self.save_csv_path, "a", newline="", encoding="utf-8") as f_csv:
            writer = csv.writer(f_csv)
            for r in results:
                writer.writerow([query, r["document"], r["entity_match"], r["completeness"],
                                 r["internal_conflict"], r["confidence_score"], r["decision"]])


# -----------------------------
# Simulation Example
# -----------------------------
if __name__ == "__main__":
    vl = ValidationLayer()

    query = "Who does Fez marry in That '70s Show?"
    top_k_docs = [
        "In the fan fiction, Fez marries Anna.",
        "In That '70s Show, Fez marries Jackie Burkhart.",
        "Fez proposes to X in season 7, but nothing else happens."
    ]

    filtered_docs, audit_results = vl.process_top_k(top_k_docs, query)

    print("\n--- Filtered Docs for Generator ---")
    for doc in filtered_docs:
        print(doc)

    print("\n--- Full Audit + Decisions ---")
    for r in audit_results:
        print(r)