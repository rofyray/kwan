"""Ghana-context practice settings for Kwan recommendations.

Each recommended skill comes with an everyday Ghanaian setting a teacher (or Syto) can use to pose
questions on that skill. Settings are grouped by NaCCA strand and by lower (B1 to B3) or upper (B4 to B6)
primary, and picked deterministically per indicator so the same skill always shows the same setting.

Round 1 uses this hand-written bank. Round 2 replaces it with exemplars extracted from the NaCCA
curriculum itself (which already uses cedi notes, cocoa and local names) plus Syto's content library.
"""

import zlib

CONTEXTS = {
    # Strand 1: Number
    (1, "lower"): [
        "Count the oranges Auntie Esi stacks in piles at her market stall, then count on as she adds more.",
        "Share 12 groundnuts equally between 3 friends after school. How many does each friend get?",
        "Use 1 cedi and 2 cedi coins to make 5 cedis in as many ways as you can.",
    ],
    (1, "upper"): [
        "A cocoa farm in the Western Region harvested 2,480 kg this season. Read the number aloud, write it in words, then round it to the nearest hundred.",
        "Kofi buys kenkey for GH₵ 7.50 and fried fish for GH₵ 12.00 and pays with a GH₵ 50 note. How much change does he get?",
        "Ama's mother cuts a pineapple into 8 equal slices and sells 3. What fraction of the pineapple is left?",
    ],
    # Strand 2: Algebra
    (2, "lower"): [
        "A kente strip repeats gold, green, gold, green. Which colour comes next? Make your own repeating pattern with bottle tops.",
        "In a game of ampe, the jumps go left, right, left, right. Clap the pattern and say what comes tenth.",
    ],
    (2, "upper"): [
        "A trotro charges GH₵ 3 for each stop. Make a table of fares for 1 to 6 stops and write the rule.",
        "Yaw saves GH₵ 2 every school day. Write a number sentence for how much he has after n days.",
    ],
    # Strand 3: Geometry and Measurement
    (3, "lower"): [
        "Find circles, squares and triangles in Adinkra symbols printed on cloth at home.",
        "Measure two sticks of sugarcane with hand spans. Which one is longer, and by how many spans?",
        "Which holds more water: a sachet of pure water or a cup? Pour and compare.",
    ],
    (3, "upper"): [
        "Measure the school garden bed with a rope and a metre stick, then work out its perimeter.",
        "A piece of kente is 4 squares long and 3 squares wide. How many squares cover it?",
        "A bus leaves Kumasi at 9:15 am and reaches Accra at 2:45 pm. How long is the journey?",
    ],
    # Strand 4: Data
    (4, "lower"): [
        "Ask your classmates their favourite food: jollof, waakye, banku or fufu. Record the answers with tally marks.",
        "Count the goats, sheep and chickens you pass on the way to school and show them in a picture graph.",
    ],
    (4, "upper"): [
        "Record how many days it rains in your town this month and draw a bar graph of each week.",
        "Toss a bottle top 20 times. Predict how often it lands face up, then compare with what happens.",
        "Survey your class on how they get to school (walking, trotro, okada, car) and find the most common way.",
    ],
}


def band(grade: int) -> str:
    """`"lower"` for Basic 1 to 3, `"upper"` for Basic 4 to 6."""
    return "lower" if grade <= 3 else "upper"


def practice_setting(indicator_id: str, strand: int, grade: int) -> str:
    """The bank setting for an indicator: `crc32(indicator_id)` picks one from its strand and band.

    Returns an empty string when the bank has nothing for that strand and band.
    """
    options = CONTEXTS.get((strand, band(grade)), [])
    if not options:
        return ""
    return options[zlib.crc32(indicator_id.encode()) % len(options)]


def bank_dict() -> dict[str, list[str]]:
    """The bank with text keys, as a bundle stores it: `{"1:lower": [...], "1:upper": [...], ...}`."""
    return {f"{strand}:{name}": list(options) for (strand, name), options in sorted(CONTEXTS.items())}
