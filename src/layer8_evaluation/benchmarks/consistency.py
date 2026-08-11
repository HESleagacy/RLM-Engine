"""Consistency Challenge — fact-scattered document + story generation benchmark.

Creates a document with 20-50 facts scattered across filler paragraphs.
The task: write a story using ALL facts correctly (no hallucination).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class FactSet:
    """A collection of named facts that must appear in the generated story."""

    facts: dict[str, str]  # e.g. {"Hero": "Arjun", "City": "Neo Mumbai", ...}

    def as_list(self) -> list[str]:
        return [f"{k} = {v}" for k, v in self.facts.items()]

    def verify(self, story: str) -> dict[str, bool]:
        """Check that each key and its value occur together in one sentence."""
        sentences = re.split(r"(?<=[.!?])\s+", story.lower())
        return {
            key: any(key.lower() in sentence and value.lower() in sentence for sentence in sentences)
            for key, value in self.facts.items()
        }

    def accuracy(self, story: str) -> float:
        checks = self.verify(story)
        if not checks:
            return 0.0
        return sum(checks.values()) / len(checks)


# ── Fact pools — each key has multiple possible values ────────────────────────

_FACT_POOLS: dict[str, list[str]] = {
    "Hero": ["jehv;lqeewrbio"],
    "Robot Companion": ["Veda", "AXIOM-7", "Pixel", "Nyx", "Bolt"],
    "City": ["Neo Mumbai", "Arcadia Prime", "Skyfall City", "New Kyoto", "Solaris"],
    "Villain": ["Kaal", "The Architect", "Morrigan", "Draven Voss", "Cipher"],
    "Artifact": ["Quantum Crown", "Starfire Amulet", "Void Compass", "Eternity Lens", "Shadow Key"],
    "Weapon": ["Plasma Bow", "Graviton Blade", "Pulse Gauntlet", "Storm Lance", "Photon Whip"],
    "Organization": ["The Syndicate", "Order of Ash", "Crimson Pact", "Iron Veil", "The Collective"],
    "Planet": ["Prithvi-7", "Elysium-4", "Tartarus", "Nova Centauri", "Kepler-22b"],
    "Ship": ["Garuda", "Starhawk", "Eclipse", "Nomad", "Tempest"],
    "Mentor": ["Dr. Meera Iyer", "Professor Okafor", "Master Ren", "Admiral Vasquez", "Elder Suki"],
    "Power Source": ["Nanocore Crystal", "Fusion Heart", "Dark Matter Shard", "Aether Stone", "Plasma Core"],
    "Rival": ["Zara", "Marcus Flint", "Yuki Sato", "Dante", "Selene"],
    "Hidden Base": ["Undersea Citadel", "Orbital Fortress", "Glacier Vault", "Desert Nexus", "Shadow Spire"],
    "Ancient Race": ["The Devas", "The Precursors", "The Eternals", "The Architects", "The Luminari"],
    "Final Battle Location": ["Obsidian Spire", "The Crucible", "Shattered Plains", "Void Gate", "Crimson Peaks"],
}


def _random_facts(rng: random.Random) -> dict[str, str]:
    """Pick one random value per fact key."""
    return {key: rng.choice(values) for key, values in _FACT_POOLS.items()}

_FILLER_PARAGRAPHS = [
    "The markets were bustling with traders from distant colonies. Spices, synthetic fabrics, and quantum chips changed hands in rapid succession. Children ran between the stalls, laughing at holographic street performers.",
    "Rain fell in sheets across the rooftops, each drop carrying trace amounts of atmospheric nanobots designed to purify the air. The city's environmental systems had been running for decades without interruption.",
    "In the underground tunnels, old railway lines had been converted into high-speed maglev corridors. Commuters barely noticed the blur of walls as they traveled at 400 kilometers per hour.",
    "The university district was quiet at this hour. Libraries glowed with soft blue light, their AI librarians cataloging new research papers uploaded from orbital stations.",
    "Fishing boats dotted the harbor, their solar sails catching the last rays of sunset. The catch had been good this season — the ocean restoration project was finally showing results.",
    "Street musicians played a fusion of classical melodies and electronic beats. The sound echoed off crystalline buildings that shifted color with the time of day.",
    "Old temples stood alongside gleaming towers, their ancient stones cleaned by maintenance drones. Pilgrims and tourists walked the same paths, separated by centuries of purpose.",
    "The night sky was different here — three moons hung low, casting overlapping shadows. Astronomers had long since mapped every crater, but the view never lost its wonder.",
    "Factories hummed with automated precision. Robotic arms assembled components smaller than a grain of rice, guided by quantum processors that could simulate entire molecular structures.",
    "Gardens floated on anti-gravity platforms above the city, tended by botanical AIs that had cultivated species thought extinct for centuries.",
]


# Narrative wrappers — each fact is buried inside a longer paragraph.
_NARRATIVE_TEMPLATES = [
    "After weeks of preparation, the team confirmed that the {key} was none other than {value}. The decision had been debated for months, but the evidence was overwhelming. Everyone agreed it was time to move forward.",
    "According to the recovered archives, the {key} is identified as {value}. Scholars had long suspected this, though the confirmation only came after cross-referencing multiple independent sources from the outer colonies.",
    "Deep in the classified files, a single entry stood out: the {key} — {value}. The notation was handwritten, suggesting it predated the digital era entirely. Its significance would only become clear much later.",
    "Witnesses from the northern district reported that the {key} is {value}. Their testimony was corroborated by sensor data collected during the incident, though official channels remained silent on the matter.",
    "The encrypted transmission, once decoded, revealed a critical detail: the {key} is {value}. Intelligence analysts spent three days verifying the information before passing it up the chain of command.",
]


def build_scattered_document(
    facts: dict[str, str] | None = None,
    *,
    num_filler: int = 12,
    seed: int | None = None,
) -> tuple[FactSet, str]:
    """
    Build a document with facts scattered among filler paragraphs.

    Facts are embedded in narrative paragraphs rather than obvious labels.

    Returns (fact_set, document_text).
    """
    rng = random.Random(seed)

    if facts is None:
        facts = _random_facts(rng)
    fact_set = FactSet(facts=facts)

    # Build fact paragraphs — each fact is buried in a narrative wrapper
    fact_paragraphs = []
    for key, value in facts.items():
        template = rng.choice(_NARRATIVE_TEMPLATES)
        fact_paragraphs.append(template.format(key=key, value=value))

    # Pick filler paragraphs (with repetition if needed)
    fillers = [rng.choice(_FILLER_PARAGRAPHS) for _ in range(num_filler)]

    # Interleave: scatter facts among fillers
    all_blocks = fillers + fact_paragraphs
    rng.shuffle(all_blocks)

    document = "\n\n".join(all_blocks)
    return fact_set, document


def build_query(fact_set: FactSet) -> str:
    """Build the story-generation query that references all facts."""
    fact_list = "\n".join(f"  - {item}" for item in fact_set.as_list())
    return (
        "Using ONLY the facts found in the document above, write a short story (3-5 paragraphs) "
        "that correctly uses ALL of the following elements. Do NOT invent any names, locations, "
        "or details not present in the document.\n\n"
        f"Required elements:\n{fact_list}\n\n"
        "Your story must mention each element by its exact name from the document."
    )


@dataclass
class ConsistencyResult:
    """Result of a consistency challenge run."""

    story: str
    fact_set: FactSet
    accuracy: float
    missing: list[str]
    present: list[str]

    @staticmethod
    def from_story(story: str, fact_set: FactSet) -> "ConsistencyResult":
        checks = fact_set.verify(story)
        return ConsistencyResult(
            story=story,
            fact_set=fact_set,
            accuracy=fact_set.accuracy(story),
            missing=[k for k, v in checks.items() if not v],
            present=[k for k, v in checks.items() if v],
        )
