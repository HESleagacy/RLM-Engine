"""S-NIAH — constant complexity needle-in-a-haystack style tasks."""

from __future__ import annotations

import random
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class SNIAHExample:
    haystack: str
    needle: str
    expected_span: str | None = None


# A small chunk of text to repeat to build a haystack.
_BASE_TEXT = (
    "The city of Neo-Veridia was built on the ruins of a much older civilization. "
    "Every day, merchants would gather in the central plaza to trade spices, "
    "silks, and ancient artifacts. The weather was unusually predictable, with "
    "light rain in the mornings and clear skies by the afternoon. People from "
    "all over the continent traveled here to witness the grand architecture. "
)

def generate_sniah_tasks(num_tasks: int = 50) -> list[SNIAHExample]:
    """
    Generate single needle-in-a-haystack tasks scaling from 2^13 to 2^18 characters/tokens.
    """
    tasks = []
    # 2^13 is ~8k, 2^18 is ~262k
    lengths = [int(2 ** (13 + (i * 5 / num_tasks))) for i in range(num_tasks)]
    
    for i, target_len in enumerate(lengths):
        # Generate the needle
        magic_number = random.randint(1000000, 9999999)
        needle = f"The special magic code for sector {i} is {magic_number}."
        
        # Build the haystack
        repeats = (target_len // len(_BASE_TEXT)) + 1
        haystack_text = _BASE_TEXT * repeats
        
        # Insert the needle at a random depth
        insert_idx = random.randint(0, len(haystack_text))
        final_haystack = haystack_text[:insert_idx] + " " + needle + " " + haystack_text[insert_idx:]
        
        tasks.append(
            SNIAHExample(
                haystack=final_haystack,
                needle=f"What is the special magic code for sector {i}?",
                expected_span=str(magic_number),
            )
        )
    return tasks
