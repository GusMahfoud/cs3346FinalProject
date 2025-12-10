import json
import os

# ============================================
# LOAD TEAM POOL (species → move names)
# ============================================
JSON_PATH = os.path.join(os.path.dirname(__file__), "teams/team_pool.json")

with open(JSON_PATH, "r") as f:
    TEAM_POOL = json.load(f)


# ============================================
# MOVE METADATA (GEN 9 UBERS / OU)
# Extend this as needed.
# ============================================
MOVE_DATA = {
    "dracometeor":   {"bp": 130, "type": "DRAGON", "category": "special"},
    "shadowball":    {"bp": 80,  "type": "GHOST",  "category": "special"},
    "flamethrower":  {"bp": 90,  "type": "FIRE",   "category": "special"},
    "uturn":         {"bp": 70,  "type": "BUG",    "category": "physical"},

    "makeitrain":    {"bp": 120, "type": "STEEL",  "category": "special"},
    "thunderbolt":   {"bp": 90,  "type": "ELECTRIC","category": "special"},
    "nastyplot":     {"bp": 0,   "type": "DARK",    "category": "status"},

    "kowtowcleave":  {"bp": 85,  "type": "DARK",    "category": "physical"},
    "suckerpunch":   {"bp": 70,  "type": "DARK",    "category": "physical"},
    "swordsdance":   {"bp": 0,   "type": "NORMAL",  "category": "status"},
    "ironhead":      {"bp": 80,  "type": "STEEL",   "category": "physical"},

    "moonblast":     {"bp": 95,  "type": "FAIRY",   "category": "special"},
    "aurasphere":    {"bp": 80,  "type": "FIGHTING","category": "special"},
    "calmmind":      {"bp": 0,   "type": "PSYCHIC", "category": "status"},

    "poisonjab":     {"bp": 80,  "type": "POISON",  "category": "physical"},
    "nightslash":    {"bp": 70,  "type": "DARK",    "category": "physical"},
    "iceshard":      {"bp": 40,  "type": "ICE",     "category": "physical"},

    "torchsong":     {"bp": 80,  "type": "FIRE",    "category": "special"},
    "slackoff":      {"bp": 0,   "type": "NORMAL",  "category": "status"},
    "willowisp":     {"bp": 0,   "type": "FIRE",    "category": "status"},
}


# ============================================
# NORMALIZE MOVE NAME → lowercase id
# ============================================
def normalize_id(name: str) -> str:
    if not name:
        return ""
    return name.lower().replace(" ", "").replace("-", "")


# ============================================
# CONVERT A MOVE NAME → FULL MOVE DICTIONARY
# ============================================
def normalize_move_dict(name: str):
    mid = normalize_id(name)

    if mid not in MOVE_DATA:
        print(f"[WARN] Missing MOVE_DATA entry for: {name} → '{mid}'")
        # fallback: treat as 0 bp physical normal
        return {
            "id": mid,
            "bp": 0,
            "type": "NORMAL",
            "category": "physical"
        }

    entry = MOVE_DATA[mid]
    return {
        "id": mid,
        "bp": entry["bp"],
        "type": entry["type"].upper(),
        "category": entry["category"].lower(),
    }


# ============================================
# GET MOVES FOR A SPECIES — NORMALIZED DICT FORM
# ============================================
def get_species_moves(species: str):
    key = species.lower()
    if key not in TEAM_POOL:
        return []

    raw_moves = TEAM_POOL[key].get("moves", [])

    normalized = [normalize_move_dict(m) for m in raw_moves]
    return normalized