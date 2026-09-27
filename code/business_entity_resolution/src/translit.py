"""Rule-based Indic -> Latin transliteration.

The Brahmic Unicode blocks (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya,
Tamil, Telugu, Kannada, Malayalam) share a parallel code-point layout: the
offset of a letter inside its 128-point block is the same across scripts.  We
therefore keep a single offset -> Latin table and apply it to every block.

No external data is used; the table is derived from the Unicode block layout.
"""

# offset inside the 0x80-wide block -> latin
_IND = {
    0x01: "n", 0x02: "n", 0x03: "h",
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l",
    0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    0x3D: "",
    # dependent vowel signs
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri",
    0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai",
    0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x5F: "", 0x60: "ri", 0x61: "li",
}
_VIRAMA = 0x4D
_NUKTA = 0x3C
# consonant offsets carry an inherent /a/
_CONS = set(range(0x15, 0x3A)) | {0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E}
# additional consonants 0x58-0x5F map onto their base letters
_EXTRA = {0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "j", 0x5C: "d",
          0x5D: "dh", 0x5E: "f"}

# start code point of every supported block
_BLOCKS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80,
           0x0C00, 0x0C80, 0x0D00, 0x0D80)


def _block_off(cp):
    """Return offset inside a supported Brahmic block, or None."""
    for b in _BLOCKS:
        if b <= cp < b + 0x80:
            return cp - b
    return None


def is_indic(s):
    for ch in s:
        cp = ord(ch)
        if 0x0900 <= cp < 0x0E00:
            return True
    return False


def translit(s):
    """Transliterate any Brahmic runs in ``s`` to Latin, leaving the rest."""
    out = []
    pending_a = False           # a consonant is waiting for its inherent /a/
    for ch in s:
        cp = ord(ch)
        off = _block_off(cp)
        if off is None:
            if pending_a:
                # word-final inherent /a/ is not pronounced (schwa deletion)
                if ch.isalnum():
                    out.append("a")
                pending_a = False
            out.append(ch)
            continue
        if off == _VIRAMA:
            pending_a = False
            continue
        if off == _NUKTA:
            continue
        if 0x66 <= off <= 0x6F:                 # script-local digits
            if pending_a:
                out.append("a")
                pending_a = False
            out.append(chr(ord("0") + off - 0x66))
            continue
        lat = _EXTRA.get(off) or _IND.get(off)
        if lat is None:
            pending_a = False
            continue
        if off in _CONS:
            if pending_a:
                out.append("a")
            out.append(lat)
            pending_a = True
        else:
            # vowel (independent or dependent) cancels the inherent /a/
            pending_a = False
            out.append(lat)
    return "".join(out)
