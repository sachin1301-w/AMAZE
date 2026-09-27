"""Text normalisation for business names and addresses.

Everything here is rule based and derived from the challenge data only; no
external gazetteer, geocoder or API is consulted.
"""
import re
import unicodedata

from translit import translit

# ---------------------------------------------------------------- basic text

_PUNCT = re.compile(r"[^0-9a-z]+")
_WS = re.compile(r"\s+")
_COMBINING = dict.fromkeys(
    c for c in range(0x300, 0x370)
)


def strip_accents(s):
    """NFKD decompose then drop combining marks: Góld -> Gold, Ópton -> Opton."""
    d = unicodedata.normalize("NFKD", s)
    if any(ord(c) >= 0x300 for c in d):
        d = "".join(c for c in d if not unicodedata.combining(c))
    return d


_AMP = re.compile(r"\s*&\s*")
_DOTCOM = re.compile(r"\b([a-z0-9][a-z0-9\-]{1,60})\.(?:com|net|org|co|in|fr|io|biz|info|us)\b")


def base_norm(s):
    """Lower-cased, accent-free, transliterated, punctuation-free token string."""
    if not s:
        return ""
    if any(ord(c) > 0x8FF for c in s):
        s = translit(s)
    s = strip_accents(s).lower()
    s = _AMP.sub(" and ", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def domain_stem(s):
    """Return the site stem when the field looks like 'foobar.com'."""
    t = strip_accents(s).lower()
    m = _DOTCOM.search(t)
    if m:
        return _PUNCT.sub("", m.group(1))
    return ""


# ------------------------------------------------- name token classification

# Legal-form tokens.  Added, dropped and re-spelled freely by the generator,
# so they carry almost no identifying signal but must be recognised to find the
# "core" of a name.  Canonical form on the right.
LEGAL = {
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd", "lim": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prvt": "pvt",
    "incorporated": "inc", "inc": "inc",
    "corporation": "corp", "corp": "corp", "corporated": "corp",
    "company": "co", "co": "co", "cie": "co",
    "llc": "llc", "l l c": "llc", "llp": "llp", "lp": "llp", "plc": "plc",
    "pllc": "llc", "pc": "pc", "pa": "pa",
    # French legal forms (the test set adds France)
    "sarl": "sarl", "sasu": "sas", "sas": "sas", "sa": "sa", "sci": "sci",
    "eurl": "eurl", "snc": "snc", "scp": "scp", "scm": "scm",
    "selarl": "selarl", "scop": "scop", "societe": "societe", "ste": "societe",
    "eirl": "eirl", "gie": "gie", "scs": "scs", "sca": "sca",
    # Indian / generic
    "opc": "opc", "huf": "huf", "assn": "assn", "association": "assn",
    "trust": "trust", "society": "society", "coop": "coop",
    "gmbh": "gmbh", "bv": "bv", "nv": "nv", "ag": "ag", "srl": "srl",
}

# Honorifics / alias markers stripped from the front of a name.
PREFIX_NOISE = {"mr", "mrs", "ms", "smt", "shri", "sri", "m", "mme", "dr",
                "fka", "f k a", "aka", "dba", "the"}

# Tokens the generator likes to *inject* into a name.  Learned offline by
# ``learn_maps.py`` (tokens that appear in a variant but never in the paired
# Source-1 name); the list below is the frozen result of that analysis.
GENERIC = set()

_ALL_LEGAL_CANON = set(LEGAL.values())


def name_tokens(nname):
    """Split a normalised name into (core tokens, legal tokens)."""
    toks = nname.split()
    while toks and toks[0] in PREFIX_NOISE:
        toks = toks[1:]
    core, legal = [], []
    for t in toks:
        if t in LEGAL:
            legal.append(LEGAL[t])
        elif len(t) == 1:
            continue
        else:
            core.append(t)
    return core, legal


# ---------------------------------------------- address token classification

# Street-type abbreviations (Rd/Road, St/Street ...).  Canonical -> short form.
STREET = {
    "street": "st", "st": "st", "str": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave",
    "drive": "dr", "dr": "dr",
    "boulevard": "blvd", "blvd": "blvd", "boul": "blvd",
    "lane": "ln", "ln": "ln",
    "court": "ct", "ct": "ct",
    "circle": "cir", "cir": "cir",
    "place": "pl", "pl": "pl",
    "terrace": "ter", "ter": "ter",
    "trail": "trl", "trl": "trl",
    "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy",
    "square": "sq", "sq": "sq",
    "expressway": "expy", "expy": "expy",
    "crossing": "xing", "xing": "xing",
    "way": "way", "walk": "walk", "run": "run", "loop": "loop",
    "point": "pt", "pt": "pt", "ridge": "rdg", "rdg": "rdg",
    "extension": "ext", "ext": "ext", "extn": "ext",
    "marg": "rd", "galli": "ln", "gali": "ln",
    # French
    "rue": "rue", "r": "rue", "boulevard_fr": "blvd", "bd": "blvd",
    "allee": "allee", "impasse": "imp", "imp": "imp",
    "chemin": "chem", "chem": "chem", "quai": "quai",
    "place_fr": "pl", "route": "rte", "rte": "rte", "avenue_fr": "ave",
}

# Unit / house-number markers that precede a number and are added at random.
UNIT_NOISE = {
    "no", "nos", "num", "number", "door", "doorno", "dno", "hno", "hn", "h",
    "flat", "plot", "shop", "unit", "apt", "apartment", "ste", "suite",
    "floor", "flr", "fl", "room", "rm", "building", "bldg", "block", "blk",
    "box", "po", "pobox", "sno", "survey", "gf", "ff", "sf", "null", "na",
    "nil", "none", "near", "opp", "opposite", "behind", "beside", "next",
    "and", "at", "the", "of", "de", "du", "des", "la", "le", "les", "a",
}

_STOP_ADDR = UNIT_NOISE | set(STREET) | set(STREET.values())


def addr_tokens(naddr):
    """Split a normalised address into (word tokens, number tokens)."""
    words, nums = [], []
    for t in naddr.split():
        if t.isdigit():
            nums.append(t)
        elif t[0].isdigit():
            # mixed like 7a / 1802b / b61 -> keep both views
            d = "".join(c for c in t if c.isdigit())
            if d:
                nums.append(d)
            w = "".join(c for c in t if not c.isdigit())
            if len(w) > 1:
                words.append(w)
        else:
            d = "".join(c for c in t if c.isdigit())
            if d:
                nums.append(d)
                w = "".join(c for c in t if not c.isdigit())
                if len(w) > 1:
                    words.append(w)
            else:
                words.append(t)
    return words, nums


def canon_addr_words(words):
    """Canonicalise street types and drop unit/filler noise."""
    out = []
    for w in words:
        if w in UNIT_NOISE:
            continue
        w = STREET.get(w, w)
        if len(w) > 1:
            out.append(w)
    return out


def strip_zeros(n):
    s = n.lstrip("0")
    return s if s else "0"
