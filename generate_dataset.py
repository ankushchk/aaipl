#!/usr/bin/env python3
"""
=============================================================================
AAIPL Fine-Tuning Dataset Generator
=============================================================================
Generates high-quality synthetic training data for all 4 competition topics:
  1. Syllogisms
  2. Seating Arrangements (Linear & Circular)
  3. Blood Relations and Family Tree
  4. Mixed Series (Alphanumeric)

Outputs:
  - data/ft_questions_<topic>.json : Q-Agent fine-tuning data (chat format)
  - data/ft_answers_<topic>.json  : A-Agent fine-tuning data (chat format)
  - data/ft_all_questions.json    : Combined Q-Agent data
  - data/ft_all_answers.json      : Combined A-Agent data
  - data/raw_questions_<topic>.json : Raw questions for testing your agents
=============================================================================
"""

import json
import random
import string
import itertools
import os
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

random.seed(42)

# ============================================================================
# TOPIC 1: SYLLOGISMS
# ============================================================================

class SyllogismGenerator:
    """Generates syllogism questions with Venn-diagram based verification."""

    CATEGORIES = [
        # (Category A, Category B) pairs for variety
        ("dogs", "animals"), ("cats", "pets"), ("roses", "flowers"),
        ("apples", "fruits"), ("cars", "vehicles"), ("novels", "books"),
        ("teachers", "professionals"), ("rivers", "water bodies"),
        ("diamonds", "gems"), ("eagles", "birds"), ("laptops", "electronics"),
        ("doctors", "graduates"), ("mangoes", "fruits"), ("tigers", "wild animals"),
        ("pianos", "instruments"), ("paintings", "artworks"), ("soldiers", "brave people"),
        ("stars", "celestial bodies"), ("tables", "furniture"), ("pens", "stationery"),
        ("whales", "mammals"), ("oaks", "trees"), ("gold", "metals"),
        ("salmon", "fish"), ("rubies", "stones"), ("trains", "transport"),
        ("poems", "literature"), ("shirts", "clothes"), ("cups", "utensils"),
        ("hills", "landforms"), ("clouds", "weather phenomena"), ("monks", "religious people"),
        ("violins", "string instruments"), ("tulips", "garden plants"), ("hawks", "raptors"),
        ("bananas", "tropical fruits"), ("wolves", "predators"), ("sparrows", "small birds"),
        ("lakes", "freshwater bodies"), ("swords", "weapons"),
    ]

    QUANTIFIERS = ["All", "Some", "No"]

    def _pick_categories(self, n=3):
        """Pick n distinct category names."""
        pool = list(self.CATEGORIES)
        random.shuffle(pool)
        names = []
        used = set()
        for a, b in pool:
            for x in (a, b):
                if x not in used:
                    names.append(x)
                    used.add(x)
                if len(names) >= n:
                    return names
        return names[:n]

    def _evaluate_syllogism(self, statements, conclusion):
        """
        Brute-force evaluate whether a conclusion MUST follow from the statements
        using a small-universe model checker.
        Returns True if the conclusion necessarily follows.
        """
        # Extract all categories mentioned
        all_cats = set()
        parsed_stmts = []
        for s in statements:
            q, a, b = self._parse_statement(s)
            parsed_stmts.append((q, a, b))
            all_cats.add(a)
            all_cats.add(b)

        cq, ca, cb = self._parse_statement(conclusion)
        all_cats.add(ca)
        all_cats.add(cb)

        cats = list(all_cats)
        # Use a universe of 4 elements
        universe = [0, 1, 2, 3]

        # Try many random assignments to find a counterexample
        for _ in range(500):
            assignment = {}
            for c in cats:
                # Random subset of universe
                assignment[c] = set(random.sample(universe, random.randint(0, 4)))

            # Check if all statements hold
            if all(self._check_statement(q, assignment[a], assignment[b]) for q, a, b in parsed_stmts):
                # Check if conclusion fails
                if not self._check_statement(cq, assignment[ca], assignment[cb]):
                    return False  # Found a counterexample
        return True

    def _parse_statement(self, stmt):
        stmt = stmt.strip().rstrip(".")
        if stmt.startswith("All "):
            rest = stmt[4:]
            parts = rest.split(" are ")
            if len(parts) != 2:
                parts = rest.split(" is ")
            return ("All", parts[0].strip(), parts[1].strip())
        elif stmt.startswith("Some "):
            rest = stmt[5:]
            parts = rest.split(" are ")
            if len(parts) != 2:
                parts = rest.split(" is ")
            return ("Some", parts[0].strip(), parts[1].strip())
        elif stmt.startswith("No "):
            rest = stmt[3:]
            parts = rest.split(" are ")
            if len(parts) != 2:
                parts = rest.split(" is ")
            return ("No", parts[0].strip(), parts[1].strip())
        return ("Some", "", "")

    def _check_statement(self, quantifier, set_a, set_b):
        if quantifier == "All":
            return set_a.issubset(set_b)
        elif quantifier == "Some":
            return len(set_a & set_b) > 0 if (set_a and set_b) else (not set_a)
        elif quantifier == "No":
            return len(set_a & set_b) == 0
        return False

    def generate(self, count=50) -> List[Dict]:
        questions = []
        attempts = 0
        while len(questions) < count and attempts < count * 20:
            attempts += 1
            q = self._generate_one()
            if q:
                questions.append(q)
        return questions

    def _generate_one(self) -> Optional[Dict]:
        cats = self._pick_categories(5)
        random.shuffle(cats)
        A, B, C, D, E = cats[:5]

        # Generate 2-3 statements
        num_stmts = random.choice([2, 3])
        if num_stmts == 2:
            groups = [(A, B), (B, C)]
        else:
            groups = [(A, B), (B, C), (C, D)]

        statements = []
        for x, y in groups:
            q = random.choice(self.QUANTIFIERS)
            statements.append(f"{q} {x} are {y}")

        # Generate 2 conclusions to test
        if num_stmts == 2:
            conc_pairs = [(A, C), (C, A), (A, B), (B, A)]
        else:
            conc_pairs = [(A, C), (A, D), (C, A), (D, A), (B, D)]

        random.shuffle(conc_pairs)
        c1_pair = conc_pairs[0]
        c2_pair = conc_pairs[1] if len(conc_pairs) > 1 else conc_pairs[0]

        c1_q = random.choice(self.QUANTIFIERS)
        c2_q = random.choice(self.QUANTIFIERS)

        conc1 = f"{c1_q} {c1_pair[0]} are {c1_pair[1]}"
        conc2 = f"{c2_q} {c2_pair[0]} are {c2_pair[1]}"

        # Evaluate conclusions
        c1_valid = self._evaluate_syllogism(statements, conc1)
        c2_valid = self._evaluate_syllogism(statements, conc2)

        if c1_valid and c2_valid:
            correct = "C"
            explanation = f"Both conclusions follow. Conclusion I ({conc1}) and Conclusion II ({conc2}) can be derived from the given statements."
        elif c1_valid:
            correct = "A"
            explanation = f"Only Conclusion I follows. {conc1} can be derived from the statements, but {conc2} cannot be definitively concluded."
        elif c2_valid:
            correct = "B"
            explanation = f"Only Conclusion II follows. {conc2} can be derived from the statements, but {conc1} cannot be definitively concluded."
        else:
            correct = "D"
            explanation = f"Neither conclusion follows. Neither {conc1} nor {conc2} can be definitively derived from the given statements."

        stmt_text = "\n".join([f"Statement {i}: {s}" for i, s in enumerate(statements, 1)])
        question_text = (
            f"{stmt_text}\n"
            f"Conclusion I: {conc1}\n"
            f"Conclusion II: {conc2}"
        )

        return {
            "topic": "Syllogisms",
            "question": question_text,
            "choices": [
                "A) If only conclusion I follows",
                "B) If only conclusion II follows",
                "C) If both conclusion I and II follow",
                "D) If neither conclusion I nor conclusion II follows"
            ],
            "answer": correct,
            "explanation": explanation
        }


# ============================================================================
# TOPIC 2: SEATING ARRANGEMENTS (Linear & Circular)
# ============================================================================

class SeatingArrangementGenerator:
    """Generates linear and circular seating arrangement puzzles."""

    NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def generate(self, count=50) -> List[Dict]:
        questions = []
        for _ in range(count):
            if random.random() < 0.5:
                q = self._generate_linear()
            else:
                q = self._generate_circular()
            if q:
                questions.append(q)
        return questions

    def _generate_linear(self) -> Dict:
        n = random.choice([5, 6, 7])
        names = self.NAMES[:n]
        arrangement = list(names)
        random.shuffle(arrangement)

        clues = self._build_linear_clues(arrangement)
        question_type = random.choice(["position", "neighbor", "between"])

        if question_type == "position":
            target = random.choice(arrangement)
            pos = arrangement.index(target) + 1
            q_text = f"{len(arrangement)} people - {', '.join(sorted(names))} - are sitting in a row facing north.\n"
            q_text += "\n".join(clues)
            q_text += f"\nWhat is the position of {target} from the left end?"

            wrong_positions = [i for i in range(1, n + 1) if i != pos]
            random.shuffle(wrong_positions)
            options_vals = [pos] + wrong_positions[:3]
            random.shuffle(options_vals)
            correct_idx = options_vals.index(pos)
            correct_letter = "ABCD"[correct_idx]
            choices = [f"{chr(65+i)}) Position {v}" for i, v in enumerate(options_vals)]
            explanation = f"{target} sits at position {pos} from the left end based on the given constraints."

        elif question_type == "neighbor":
            idx = random.randint(1, n - 2)  # not endpoints
            target = arrangement[idx]
            left_n = arrangement[idx - 1]
            right_n = arrangement[idx + 1]

            q_text = f"{len(arrangement)} people - {', '.join(sorted(names))} - are sitting in a row facing north.\n"
            q_text += "\n".join(clues)
            q_text += f"\nWho are the immediate neighbors of {target}?"

            correct_ans = f"{left_n} and {right_n}"
            wrong_answers = []
            others = [x for x in names if x != target and x != left_n and x != right_n]
            for _ in range(3):
                if len(others) >= 2:
                    pair = random.sample(others, 2)
                    wrong_answers.append(f"{pair[0]} and {pair[1]}")
                else:
                    wrong_answers.append(f"{random.choice(others)} and {target}")

            options = [correct_ans] + wrong_answers[:3]
            random.shuffle(options)
            correct_idx = options.index(correct_ans)
            correct_letter = "ABCD"[correct_idx]
            choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]
            explanation = f"Based on the arrangement, {target}'s immediate neighbors are {left_n} (left) and {right_n} (right)."

        else:  # between
            if n < 4:
                return self._generate_linear()
            idx = random.randint(1, n - 2)
            target = arrangement[idx]
            left_n = arrangement[idx - 1]
            right_n = arrangement[idx + 1]

            q_text = f"{len(arrangement)} people - {', '.join(sorted(names))} - are sitting in a row facing north.\n"
            q_text += "\n".join(clues)
            q_text += f"\nWho sits between {left_n} and {right_n}?"

            others = [x for x in names if x != target and x != left_n and x != right_n]
            options = [target] + random.sample(others, min(3, len(others)))
            while len(options) < 4:
                options.append(random.choice(others))
            random.shuffle(options)
            correct_idx = options.index(target)
            correct_letter = "ABCD"[correct_idx]
            choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]
            explanation = f"{target} sits between {left_n} and {right_n} based on the given constraints."

        return {
            "topic": "Seating Arrangements (Linear, Circular)",
            "question": q_text,
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _build_linear_clues(self, arrangement: List[str]) -> List[str]:
        n = len(arrangement)
        clues = []
        revealed = set()

        # Clue 1: endpoint
        end = random.choice([0, n - 1])
        side = "left" if end == 0 else "right"
        clues.append(f"{arrangement[end]} sits at the {side} end.")
        revealed.add(arrangement[end])

        # Clue 2-3: relative positions
        for _ in range(random.randint(2, 3)):
            idx = random.randint(0, n - 2)
            p1, p2 = arrangement[idx], arrangement[idx + 1]
            clue_type = random.choice(["immediate_right", "immediate_left"])
            if clue_type == "immediate_right":
                clues.append(f"{p2} sits immediately to the right of {p1}.")
            else:
                clues.append(f"{p1} sits immediately to the left of {p2}.")
            revealed.add(p1)
            revealed.add(p2)

        # Clue 3: gap clue
        if n >= 5:
            gap = random.randint(2, min(3, n - 2))
            idx1 = random.randint(0, n - 1 - gap)
            p1 = arrangement[idx1]
            p2 = arrangement[idx1 + gap]
            clues.append(f"There {'is 1 person' if gap == 2 else f'are {gap - 1} people'} sitting between {p1} and {p2}.")
            revealed.add(p1)
            revealed.add(p2)

        return clues

    def _generate_circular(self) -> Dict:
        n = random.choice([5, 6, 7, 8])
        names = self.NAMES[:n]
        arrangement = list(names)
        random.shuffle(arrangement)

        clues = self._build_circular_clues(arrangement)
        target_idx = random.randint(0, n - 1)
        target = arrangement[target_idx]
        left_n = arrangement[(target_idx - 1) % n]
        right_n = arrangement[(target_idx + 1) % n]
        opposite = arrangement[(target_idx + n // 2) % n] if n % 2 == 0 else None

        q_text = f"{n} people - {', '.join(sorted(names))} - are sitting around a circular table facing the center.\n"
        q_text += "\n".join(clues)

        question_subtype = random.choice(["neighbor_right", "neighbor_left", "opposite"] if opposite else ["neighbor_right", "neighbor_left"])

        if question_subtype == "neighbor_right":
            q_text += f"\nWho is sitting to the immediate right of {target}?"
            correct_ans = right_n
            explanation = f"Based on the circular arrangement, {right_n} sits immediately to the right of {target}."
        elif question_subtype == "neighbor_left":
            q_text += f"\nWho is sitting to the immediate left of {target}?"
            correct_ans = left_n
            explanation = f"Based on the circular arrangement, {left_n} sits immediately to the left of {target}."
        else:
            q_text += f"\nWho sits directly opposite to {target}?"
            correct_ans = opposite
            explanation = f"In the circular arrangement with {n} people, {opposite} sits directly opposite to {target}."

        others = [x for x in names if x != correct_ans and x != target]
        options = [correct_ans] + random.sample(others, min(3, len(others)))
        while len(options) < 4:
            options.append(random.choice(others))
        random.shuffle(options)
        correct_idx = options.index(correct_ans)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        return {
            "topic": "Seating Arrangements (Linear, Circular)",
            "question": q_text,
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _build_circular_clues(self, arrangement: List[str]) -> List[str]:
        n = len(arrangement)
        clues = []

        for _ in range(random.randint(3, 4)):
            idx = random.randint(0, n - 1)
            right_idx = (idx + 1) % n
            left_idx = (idx - 1) % n
            p = arrangement[idx]
            clue_type = random.choice(["right", "left", "between"])

            if clue_type == "right":
                clues.append(f"{arrangement[right_idx]} sits to the immediate right of {p}.")
            elif clue_type == "left":
                clues.append(f"{arrangement[left_idx]} sits to the immediate left of {p}.")
            else:
                clues.append(f"{p} sits between {arrangement[left_idx]} and {arrangement[right_idx]}.")

        if n % 2 == 0:
            idx = random.randint(0, n - 1)
            opp_idx = (idx + n // 2) % n
            clues.append(f"{arrangement[idx]} sits directly opposite to {arrangement[opp_idx]}.")

        return clues


# ============================================================================
# TOPIC 3: BLOOD RELATIONS AND FAMILY TREE
# ============================================================================

class BloodRelationGenerator:
    """Generates blood relation and family tree puzzles."""

    MALE_NAMES = [
        "Arjun", "Ravi", "Suresh", "Mukesh", "Ramesh", "Pradeep", "Vikram",
        "Anil", "Deepak", "Sanjay", "Rahul", "Nitin", "Ajay", "Manoj",
        "Gopal", "Kiran", "Rohit", "Amit", "Vishal", "Tarun",
        "David", "John", "Peter", "Paul", "Mark", "James", "Robert",
        "Tom", "Sam", "Jack", "Leo", "Max", "Ben", "Dan", "Ray"
    ]

    FEMALE_NAMES = [
        "Sita", "Neha", "Priya", "Asha", "Meera", "Lakshmi", "Radha",
        "Kavita", "Sunita", "Rekha", "Anita", "Geeta", "Nisha", "Pooja",
        "Swati", "Renu", "Deepa", "Shanti", "Usha", "Veena",
        "Mary", "Sarah", "Lisa", "Emma", "Anna", "Jane", "Helen",
        "Rose", "Amy", "Lucy", "Kate", "Zoe", "Eva", "Ivy", "Mia"
    ]

    # Relationship chains and their resolutions
    RELATION_CHAINS = [
        # (chain_description_template, answer_relation, needs_gender)
        # Male speaker
        ("{speaker} says, 'She is the wife of my father's only son.'", "Wife", "male"),
        ("{speaker} says, 'He is the son of my mother's husband.'", "Brother", "male"),
        ("{speaker} says, 'She is the daughter of my grandfather's only son.'", "Sister", "male"),
        ("{speaker} says, 'He is the father of my sister's son.'", "Brother-in-law", "male"),
        ("{speaker} says, 'She is the mother of my father's grandchild.'", "Mother or Sister-in-law", "male"),
        ("{speaker} says, 'He is the son of my father's father.'", "Father or Uncle", "male"),
        # Female speaker
        ("{speaker} says, 'He is the husband of my mother's only daughter.'", "Husband", "female"),
        ("{speaker} says, 'She is the wife of my father's only son.'", "Sister-in-law or Self (if married to father's son)", "female"),
        ("{speaker} says, 'He is the only son of my grandfather.'", "Father", "female"),
        ("{speaker} says, 'She is the daughter of my mother's husband.'", "Sister", "female"),
    ]

    # Simpler, more standard relation puzzles with guaranteed single answers
    TEMPLATES = [
        {
            "template": "Pointing to a photograph, {A} said, 'He is the son of the only son of my father.' How is the person in the photograph related to {A}?",
            "answer": "Son",
            "distractors": ["Nephew", "Brother", "Grandson"],
            "explanation": "{A}'s father's only son is {A} himself. Therefore, the person in the photograph is {A}'s son."
        },
        {
            "template": "Pointing to a woman, {A} said, 'She is the daughter of the only child of my grandmother.' How is the woman related to {A}?",
            "answer": "Sister",
            "distractors": ["Mother", "Daughter", "Aunt"],
            "explanation": "The only child of {A}'s grandmother is {A}'s parent. The daughter of {A}'s parent is {A}'s sister."
        },
        {
            "template": "{A} said, 'This girl is the wife of the grandson of my mother.' How is {A} related to the girl?",
            "answer": "Father-in-law",
            "distractors": ["Grandfather", "Uncle", "Brother-in-law"],
            "explanation": "The grandson of {A}'s mother is {A}'s son. The wife of {A}'s son is {A}'s daughter-in-law. So {A} is the girl's father-in-law."
        },
        {
            "template": "Introducing a man, {A} said, 'He is the only son of the father of my mother.' How is the man related to {A}?",
            "answer": "Maternal uncle",
            "distractors": ["Father", "Grandfather", "Brother"],
            "explanation": "The father of {A}'s mother is {A}'s maternal grandfather. The only son of {A}'s maternal grandfather is {A}'s maternal uncle."
        },
        {
            "template": "Looking at a portrait of a man, {A} said, 'His mother is the wife of my father's son. I have no brothers or sisters.' Whose portrait was {A} looking at?",
            "answer": "His son",
            "distractors": ["His father", "His nephew", "His cousin"],
            "explanation": "Since {A} has no brothers or sisters, 'my father's son' is {A} himself. The wife of {A} is {A}'s wife. The man whose mother is {A}'s wife must be {A}'s son."
        },
        {
            "template": "Pointing to a lady, {A} said, 'She is the sister of the father of my nephew.' How is the lady related to {A}?",
            "answer": "Sister",
            "distractors": ["Mother", "Aunt", "Sister-in-law"],
            "explanation": "The father of {A}'s nephew is {A}'s brother. The sister of {A}'s brother is {A}'s sister."
        },
        {
            "template": "{A} and {B} are children of {C}. {C} is the mother of {A} but {A} is not the daughter of {C}. What is the relationship between {A} and {B}?",
            "answer": "{A} is the brother of {B}",
            "distractors": ["{A} is the sister of {B}", "{A} is the father of {B}", "{A} is the cousin of {B}"],
            "explanation": "Since {C} is {A}'s mother but {A} is not {C}'s daughter, {A} must be {C}'s son, i.e., male. So {A} is {B}'s brother."
        },
        {
            "template": "If {A} says, '{B} is the brother of the son of my grandfather's only child,' how is {B} related to {A}?",
            "answer": "Brother",
            "distractors": ["Cousin", "Uncle", "Father"],
            "explanation": "{A}'s grandfather's only child is {A}'s parent. The son of {A}'s parent is {A}'s brother. The brother of {A}'s brother is also {A}'s brother. So {B} is {A}'s brother."
        },
        {
            "template": "Introducing a girl, {A} says, 'She is the only daughter of the father of the mother of my sister.' How is the girl related to {A}?",
            "answer": "Mother",
            "distractors": ["Sister", "Aunt", "Grandmother"],
            "explanation": "The mother of {A}'s sister is {A}'s mother. The father of {A}'s mother is {A}'s maternal grandfather. The only daughter of the maternal grandfather is {A}'s mother."
        },
        {
            "template": "{A} pointing to {B} says, '{B} is the son of my father's brother.' How is {B} related to {A}?",
            "answer": "Cousin",
            "distractors": ["Brother", "Nephew", "Uncle"],
            "explanation": "{A}'s father's brother is {A}'s uncle. The son of {A}'s uncle is {A}'s cousin."
        },
        {
            "template": "Pointing to a photograph, {A} said, 'She is the granddaughter of the elder brother of my father.' How is the girl in the photograph related to {A}?",
            "answer": "Cousin",
            "distractors": ["Niece", "Sister", "Daughter"],
            "explanation": "The elder brother of {A}'s father is {A}'s uncle. The granddaughter of {A}'s uncle is {A}'s cousin (once removed) or simply cousin."
        },
        {
            "template": "Deepa's mother is the only daughter of {A}'s mother. How is {A} related to Deepa?",
            "answer": "Maternal uncle",
            "distractors": ["Father", "Grandfather", "Brother"],
            "explanation": "The only daughter of {A}'s mother is {A}'s sister. If Deepa's mother is {A}'s sister, then {A} is Deepa's maternal uncle."
        },
        {
            "template": "{A} is the son of {C}. {C} is the daughter of {B}. {D} is the son of {B}. How is {D} related to {A}?",
            "answer": "Uncle",
            "distractors": ["Father", "Brother", "Grandfather"],
            "explanation": "{C} is {A}'s mother and {B}'s daughter. {D} is {B}'s son, making {D} the brother of {C}. The brother of one's mother is one's uncle."
        },
        {
            "template": "If '{A} is the brother of {B}', '{B} is the sister of {C}', and '{C} is the father of {D}', how is {A} related to {D}?",
            "answer": "Uncle",
            "distractors": ["Father", "Grandfather", "Cousin"],
            "explanation": "{A} is {B}'s brother, {B} is {C}'s sister (so {A} and {C} are siblings), and {C} is {D}'s father. Therefore {A} is {D}'s uncle."
        },
        {
            "template": "Pointing to a man in a photograph, {A} said, 'The only daughter of his mother is my mother.' How is {A} related to the man?",
            "answer": "Nephew or Niece",
            "distractors": ["Son", "Grandson", "Cousin"],
            "explanation": "The only daughter of the man's mother is the man's sister. If the man's sister is {A}'s mother, then the man is {A}'s maternal uncle. So {A} is the man's nephew/niece."
        },
    ]

    def generate(self, count=50) -> List[Dict]:
        questions = []
        for i in range(count):
            q = self._generate_one(i)
            if q:
                questions.append(q)
        return questions

    def _generate_one(self, seed_idx) -> Dict:
        template = self.TEMPLATES[seed_idx % len(self.TEMPLATES)]
        # Pick random names
        males = random.sample(self.MALE_NAMES, 4)
        females = random.sample(self.FEMALE_NAMES, 2)

        name_map = {
            "A": males[0], "B": males[1], "C": females[0] if "mother" in template["template"].lower() or "daughter" in template["template"].lower() else males[2],
            "D": males[3] if len(males) > 3 else females[1]
        }

        question_text = template["template"]
        answer = template["answer"]
        explanation = template["explanation"]
        distractors = list(template["distractors"])

        for key, name in name_map.items():
            question_text = question_text.replace("{" + key + "}", name)
            answer = answer.replace("{" + key + "}", name)
            explanation = explanation.replace("{" + key + "}", name)
            distractors = [d.replace("{" + key + "}", name) for d in distractors]

        options = [answer] + distractors[:3]
        random.shuffle(options)
        correct_idx = options.index(answer)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        return {
            "topic": "Blood Relations and Family Tree",
            "question": question_text,
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }


# ============================================================================
# TOPIC 4: MIXED SERIES (ALPHANUMERIC)
# ============================================================================

class AlphanumericSeriesGenerator:
    """Generates alphanumeric series / pattern completion questions."""

    def generate(self, count=50) -> List[Dict]:
        questions = []
        generators = [
            self._letter_shift_series,
            self._number_letter_mix,
            self._triple_letter_pattern,
            self._alternating_pattern,
            self._reverse_alphabet_series,
            self._number_series_with_letters,
            self._position_based_series,
            self._mirror_series,
        ]
        for i in range(count):
            gen = generators[i % len(generators)]
            q = gen()
            if q:
                questions.append(q)
        return questions

    def _letter_at(self, pos):
        """Get letter at position (0=A, 25=Z), with wrapping."""
        return chr(65 + (pos % 26))

    def _letter_shift_series(self) -> Dict:
        """Series like: JAK, KBL, LCM, MDN, ___"""
        start1 = random.randint(0, 20)
        start2 = random.randint(0, 20)
        start3 = random.randint(0, 20)
        shift1 = random.randint(1, 2)
        shift2 = random.randint(1, 2)
        shift3 = random.randint(1, 2)
        n_terms = random.choice([4, 5])

        terms = []
        for i in range(n_terms + 1):
            t = self._letter_at(start1 + i * shift1) + self._letter_at(start2 + i * shift2) + self._letter_at(start3 + i * shift3)
            terms.append(t)

        shown = ", ".join(terms[:-1])
        correct = terms[-1]

        # Generate distractors
        distractors = set()
        for _ in range(20):
            d = ""
            for c in correct:
                offset = random.choice([-2, -1, 1, 2])
                d += chr(65 + (ord(c) - 65 + offset) % 26)
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            d = "".join(random.choice(string.ascii_uppercase) for _ in range(3))
            if d != correct and d not in distractors:
                distractors.append(d)

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"Each letter in the triplet follows a consistent pattern. Applying the same shifts to the last term gives {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"What comes next in the series: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _number_letter_mix(self) -> Dict:
        """Series like: A2, C4, E6, G8, ___"""
        start_letter = random.randint(0, 15)
        letter_step = random.choice([1, 2, 3])
        start_num = random.randint(1, 5)
        num_step = random.choice([1, 2, 3])
        n_terms = random.choice([4, 5])

        terms = []
        for i in range(n_terms + 1):
            l = self._letter_at(start_letter + i * letter_step)
            n = start_num + i * num_step
            terms.append(f"{l}{n}")

        shown = ", ".join(terms[:-1])
        correct = terms[-1]

        distractors = set()
        for _ in range(20):
            dl = self._letter_at(start_letter + n_terms * letter_step + random.choice([-1, 1, 2]))
            dn = start_num + n_terms * num_step + random.choice([-1, 1, 2])
            d = f"{dl}{dn}"
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            d = f"{random.choice(string.ascii_uppercase)}{random.randint(1, 20)}"
            if d != correct and d not in distractors:
                distractors.append(d)

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter_ans = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"Letters increase by {letter_step} positions and numbers increase by {num_step}. Next term: {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"Find the next term in the series: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter_ans,
            "explanation": explanation
        }

    def _triple_letter_pattern(self) -> Dict:
        """Series like: BCB, DED, FGF, HIH, ___"""
        start = random.randint(0, 15)
        step = random.choice([2, 3])
        n_terms = random.choice([4, 5])

        terms = []
        for i in range(n_terms + 1):
            base = start + i * step
            t = self._letter_at(base) + self._letter_at(base + 1) + self._letter_at(base)
            terms.append(t)

        shown = ", ".join(terms[:-1])
        correct = terms[-1]

        distractors = set()
        for _ in range(20):
            base = start + n_terms * step + random.choice([-1, 1])
            d = self._letter_at(base) + self._letter_at(base + 1) + self._letter_at(base)
            if d != correct:
                distractors.add(d)
            # Also try wrong patterns
            base = start + n_terms * step
            d = self._letter_at(base + 1) + self._letter_at(base) + self._letter_at(base + 1)
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            d = "".join(random.choice(string.ascii_uppercase) for _ in range(3))
            if d != correct and d not in distractors:
                distractors.append(d)

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"First two letters progress alphabetically in pairs with step {step}. Third letter repeats the first. Next: {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"What comes next: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _alternating_pattern(self) -> Dict:
        """Series with alternating elements: A1B, C2D, E3F, ___"""
        start_l = random.randint(0, 15)
        start_n = random.randint(1, 5)
        l_step = random.choice([2, 3])
        n_terms = random.choice([4, 5])

        terms = []
        for i in range(n_terms + 1):
            l1 = self._letter_at(start_l + i * l_step)
            n = start_n + i
            l2 = self._letter_at(start_l + i * l_step + 1)
            terms.append(f"{l1}{n}{l2}")

        shown = ", ".join(terms[:-1])
        correct = terms[-1]

        distractors = set()
        for _ in range(20):
            offset = random.choice([-1, 1, 2])
            l1 = self._letter_at(start_l + n_terms * l_step + offset)
            n = start_n + n_terms + random.choice([-1, 0, 1])
            l2 = self._letter_at(start_l + n_terms * l_step + 1 + offset)
            d = f"{l1}{n}{l2}"
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            d = f"{random.choice(string.ascii_uppercase)}{random.randint(1,9)}{random.choice(string.ascii_uppercase)}"
            if d != correct and d not in distractors:
                distractors.append(d)

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"Letters increase by {l_step} positions, numbers increase by 1. Next term: {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"Complete the series: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _reverse_alphabet_series(self) -> Dict:
        """Z, Y, X, W... mixed with numbers or forward series"""
        start_fwd = random.randint(0, 10)
        start_rev = random.randint(20, 25)
        n_terms = random.choice([4, 5, 6])

        terms = []
        for i in range(n_terms + 1):
            fwd = self._letter_at(start_fwd + i)
            rev = self._letter_at(start_rev - i)
            terms.append(f"{rev}{fwd}")

        shown = ", ".join(terms[:-1])
        correct = terms[-1]

        distractors = set()
        for _ in range(20):
            offset = random.choice([-1, 1, 2])
            d = self._letter_at(start_rev - n_terms + offset) + self._letter_at(start_fwd + n_terms + offset)
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            d = "".join(random.choice(string.ascii_uppercase) for _ in range(2))
            if d != correct and d not in distractors:
                distractors.append(d)

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"First letter decreases (reverse alphabet) while second letter increases (forward alphabet). Next: {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"What comes next in the series: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _number_series_with_letters(self) -> Dict:
        """Pure number series like: 2, 5, 11, 23, ___"""
        pattern_type = random.choice(["multiply_add", "add_increasing", "square_based"])

        if pattern_type == "multiply_add":
            start = random.randint(1, 5)
            mult = random.choice([2, 3])
            add = random.choice([1, -1, 2, 3])
            terms = [start]
            for _ in range(5):
                terms.append(terms[-1] * mult + add)
        elif pattern_type == "add_increasing":
            start = random.randint(1, 10)
            base_diff = random.randint(1, 5)
            diff_inc = random.choice([1, 2, 3])
            terms = [start]
            diff = base_diff
            for _ in range(5):
                terms.append(terms[-1] + diff)
                diff += diff_inc
        else:  # square_based
            start = random.randint(1, 4)
            terms = []
            for i in range(6):
                terms.append((start + i) ** 2 + random.choice([0, 1]))

        n_show = random.choice([4, 5])
        shown = ", ".join(str(t) for t in terms[:n_show])
        correct = str(terms[n_show])

        distractors = set()
        for _ in range(20):
            d = str(terms[n_show] + random.choice([-3, -2, -1, 1, 2, 3]))
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"Following the pattern of the series, the next term is {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"Find the next number in the series: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _position_based_series(self) -> Dict:
        """Series based on letter positions: A1Z, B2Y, C3X, ___"""
        n_terms = random.choice([4, 5])
        terms = []
        for i in range(n_terms + 1):
            fwd = self._letter_at(i)
            num = i + 1
            rev = self._letter_at(25 - i)
            terms.append(f"{fwd}{num}{rev}")

        shown = ", ".join(terms[:-1])
        correct = terms[-1]

        distractors = set()
        for _ in range(20):
            offset = random.choice([-1, 1, 2])
            d = self._letter_at(n_terms + offset) + str(n_terms + 1 + offset) + self._letter_at(25 - n_terms + offset)
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            d = f"{random.choice(string.ascii_uppercase)}{random.randint(1,9)}{random.choice(string.ascii_uppercase)}"
            if d != correct and d not in distractors:
                distractors.append(d)

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"First letter goes forward (A,B,C...), number increases by 1, last letter goes backward (Z,Y,X...). Next: {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"What comes next: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }

    def _mirror_series(self) -> Dict:
        """Series with mirror/palindrome pattern: ABA, BCB, CDC, ___"""
        start = random.randint(0, 18)
        step = random.choice([1, 2])
        n_terms = random.choice([4, 5])

        terms = []
        for i in range(n_terms + 1):
            base = start + i * step
            t = self._letter_at(base) + self._letter_at(base + 1) + self._letter_at(base)
            terms.append(t)

        shown = ", ".join(terms[:-1])
        correct = terms[-1]

        distractors = set()
        for _ in range(20):
            base = start + n_terms * step + random.choice([-1, 1])
            d = self._letter_at(base) + self._letter_at(base + 1) + self._letter_at(base)
            if d != correct:
                distractors.add(d)
        distractors = list(distractors)[:3]
        while len(distractors) < 3:
            d = "".join(random.choice(string.ascii_uppercase) for _ in range(3))
            if d != correct and d not in distractors:
                distractors.append(d)

        options = [correct] + distractors
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCD"[correct_idx]
        choices = [f"{chr(65+i)}) {v}" for i, v in enumerate(options)]

        explanation = f"Each term is a palindrome where the outer letters match. Letters advance by {step}. Next: {correct}."

        return {
            "topic": "Mixed Series (Alphanumeric)",
            "question": f"Complete the series: {shown}, ___?",
            "choices": choices,
            "answer": correct_letter,
            "explanation": explanation
        }


# ============================================================================
# DATASET FORMATTER - Converts to fine-tuning chat format
# ============================================================================

class DatasetFormatter:
    """Converts raw Q&A into fine-tuning format for Q-Agent and A-Agent."""

    @staticmethod
    def format_q_agent_data(questions: List[Dict], topic_name: str) -> List[Dict]:
        """
        Format data for Q-Agent fine-tuning.
        The model learns to GENERATE questions given a topic.
        """
        ft_data = []
        for q in questions:
            # Build the user prompt (what the Q-agent receives)
            user_msg = (
                f"Generate an EXTREMELY DIFFICULT MCQ on the topic: {q['topic']}.\n"
                f"Return your response as a valid JSON object with keys: topic, question, choices, answer, explanation."
            )

            # Build the assistant response (what the Q-agent should produce)
            assistant_msg = json.dumps({
                "topic": q["topic"],
                "question": q["question"],
                "choices": q["choices"],
                "answer": q["answer"],
                "explanation": q["explanation"]
            }, indent=2)

            ft_data.append({
                "conversations": [
                    {"role": "system", "content": "You are an expert-level examiner creating extremely challenging multiple-choice questions for competitive exams. Generate questions in valid JSON format."},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            })
        return ft_data

    @staticmethod
    def format_a_agent_data(questions: List[Dict]) -> List[Dict]:
        """
        Format data for A-Agent fine-tuning.
        The model learns to ANSWER questions correctly.
        """
        ft_data = []
        for q in questions:
            choices_str = " ".join(q["choices"])
            user_msg = (
                f"Question: {q['question']}\n"
                f"Choices: {choices_str}\n\n"
                f"Answer the question. Return a valid JSON with keys: answer, reasoning."
            )

            assistant_msg = json.dumps({
                "answer": q["answer"],
                "reasoning": q.get("explanation", "Based on careful analysis of the given information.")
            }, indent=2)

            ft_data.append({
                "conversations": [
                    {"role": "system", "content": "You are an expert in quantitative aptitude for competitive exams. Solve MCQs with step-by-step reasoning. Return your answer as valid JSON."},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            })
        return ft_data

    @staticmethod
    def format_raw_questions(questions: List[Dict]) -> List[Dict]:
        """Format as raw questions.json (for testing with your agents)."""
        return [{
            "topic": q["topic"],
            "question": q["question"],
            "choices": q["choices"],
            "answer": q["answer"],
            "explanation": q.get("explanation", "")
        } for q in questions]


# ============================================================================
# MAIN - Generate everything
# ============================================================================

def main():
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Configuration: how many questions per topic
    COUNTS = {
        "syllogisms": 60,
        "seating": 60,
        "blood_relations": 60,
        "alphanumeric": 60,
    }

    print("=" * 70)
    print("🚀 AAIPL Fine-Tuning Dataset Generator")
    print("=" * 70)

    formatter = DatasetFormatter()

    all_q_agent_data = []
    all_a_agent_data = []
    all_raw_questions = []

    # ---- Topic 1: Syllogisms ----
    print(f"\n📝 Generating {COUNTS['syllogisms']} Syllogism questions...")
    syl_gen = SyllogismGenerator()
    syl_questions = syl_gen.generate(COUNTS["syllogisms"])
    print(f"   ✅ Generated {len(syl_questions)} valid syllogism questions")

    q_data = formatter.format_q_agent_data(syl_questions, "Syllogisms")
    a_data = formatter.format_a_agent_data(syl_questions)
    raw = formatter.format_raw_questions(syl_questions)

    save_json(q_data, output_dir / "ft_questions_syllogisms.json")
    save_json(a_data, output_dir / "ft_answers_syllogisms.json")
    save_json(raw, output_dir / "raw_questions_syllogisms.json")
    all_q_agent_data.extend(q_data)
    all_a_agent_data.extend(a_data)
    all_raw_questions.extend(raw)

    # ---- Topic 2: Seating Arrangements ----
    print(f"\n📝 Generating {COUNTS['seating']} Seating Arrangement questions...")
    seat_gen = SeatingArrangementGenerator()
    seat_questions = seat_gen.generate(COUNTS["seating"])
    print(f"   ✅ Generated {len(seat_questions)} valid seating arrangement questions")

    q_data = formatter.format_q_agent_data(seat_questions, "Seating Arrangements (Linear, Circular)")
    a_data = formatter.format_a_agent_data(seat_questions)
    raw = formatter.format_raw_questions(seat_questions)

    save_json(q_data, output_dir / "ft_questions_seating.json")
    save_json(a_data, output_dir / "ft_answers_seating.json")
    save_json(raw, output_dir / "raw_questions_seating.json")
    all_q_agent_data.extend(q_data)
    all_a_agent_data.extend(a_data)
    all_raw_questions.extend(raw)

    # ---- Topic 3: Blood Relations ----
    print(f"\n📝 Generating {COUNTS['blood_relations']} Blood Relations questions...")
    blood_gen = BloodRelationGenerator()
    blood_questions = blood_gen.generate(COUNTS["blood_relations"])
    print(f"   ✅ Generated {len(blood_questions)} valid blood relation questions")

    q_data = formatter.format_q_agent_data(blood_questions, "Blood Relations and Family Tree")
    a_data = formatter.format_a_agent_data(blood_questions)
    raw = formatter.format_raw_questions(blood_questions)

    save_json(q_data, output_dir / "ft_questions_blood_relations.json")
    save_json(a_data, output_dir / "ft_answers_blood_relations.json")
    save_json(raw, output_dir / "raw_questions_blood_relations.json")
    all_q_agent_data.extend(q_data)
    all_a_agent_data.extend(a_data)
    all_raw_questions.extend(raw)

    # ---- Topic 4: Alphanumeric Series ----
    print(f"\n📝 Generating {COUNTS['alphanumeric']} Alphanumeric Series questions...")
    alpha_gen = AlphanumericSeriesGenerator()
    alpha_questions = alpha_gen.generate(COUNTS["alphanumeric"])
    print(f"   ✅ Generated {len(alpha_questions)} valid alphanumeric series questions")

    q_data = formatter.format_q_agent_data(alpha_questions, "Mixed Series (Alphanumeric)")
    a_data = formatter.format_a_agent_data(alpha_questions)
    raw = formatter.format_raw_questions(alpha_questions)

    save_json(q_data, output_dir / "ft_questions_alphanumeric.json")
    save_json(a_data, output_dir / "ft_answers_alphanumeric.json")
    save_json(raw, output_dir / "raw_questions_alphanumeric.json")
    all_q_agent_data.extend(q_data)
    all_a_agent_data.extend(a_data)
    all_raw_questions.extend(raw)

    # ---- Combined files ----
    random.shuffle(all_q_agent_data)
    random.shuffle(all_a_agent_data)
    random.shuffle(all_raw_questions)

    save_json(all_q_agent_data, output_dir / "ft_all_questions.json")
    save_json(all_a_agent_data, output_dir / "ft_all_answers.json")
    save_json(all_raw_questions, output_dir / "raw_all_questions.json")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("🎉 DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 Total questions generated: {len(all_raw_questions)}")
    print(f"   • Syllogisms:          {len(syl_questions)}")
    print(f"   • Seating Arrangements:{len(seat_questions)}")
    print(f"   • Blood Relations:     {len(blood_questions)}")
    print(f"   • Alphanumeric Series: {len(alpha_questions)}")

    print(f"\n📁 Output files in '{output_dir}/':")
    print(f"   Q-Agent Fine-Tuning:")
    print(f"     • ft_questions_syllogisms.json     ({len(syl_questions)} samples)")
    print(f"     • ft_questions_seating.json         ({len(seat_questions)} samples)")
    print(f"     • ft_questions_blood_relations.json ({len(blood_questions)} samples)")
    print(f"     • ft_questions_alphanumeric.json    ({len(alpha_questions)} samples)")
    print(f"     • ft_all_questions.json             ({len(all_q_agent_data)} samples)")
    print(f"   A-Agent Fine-Tuning:")
    print(f"     • ft_answers_syllogisms.json       ({len(syl_questions)} samples)")
    print(f"     • ft_answers_seating.json           ({len(seat_questions)} samples)")
    print(f"     • ft_answers_blood_relations.json   ({len(blood_questions)} samples)")
    print(f"     • ft_answers_alphanumeric.json      ({len(alpha_questions)} samples)")
    print(f"     • ft_all_answers.json               ({len(all_a_agent_data)} samples)")
    print(f"   Raw Questions (for testing):")
    print(f"     • raw_all_questions.json            ({len(all_raw_questions)} samples)")

    print(f"\n💡 Next steps:")
    print(f"   1. Use 'ft_all_questions.json' to fine-tune your Q-Agent")
    print(f"   2. Use 'ft_all_answers.json' to fine-tune your A-Agent")
    print(f"   3. Use 'raw_all_questions.json' to test your agents locally")
    print(f"   4. Follow the Unsloth tutorial in tutorial.ipynb for fine-tuning")


def save_json(data, path):
    """Save data as pretty-printed JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"   💾 Saved: {path} ({len(data)} items)")


if __name__ == "__main__":
    main()
