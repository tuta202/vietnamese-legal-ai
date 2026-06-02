"""
LegalTextParser — converts raw .txt files from thuvienphapluat.vn into
structured Article objects.

Text format: each raw file is a continuous stream where HTML line-wrapping
creates arbitrary newlines, including mid-word breaks. Strategy:
  1. Normalise all whitespace to single space.
  2. Find article-header Điều markers using regex + negative lookahead.
  3. Apply a monotonic-sequence filter to discard in-text cross-references.
"""
import re
from dataclasses import dataclass, field, asdict
from hashlib import md5
from typing import Optional

# ---------------------------------------------------------------------------
# Regex: Điều + number (+ optional letter) + period/colon + space
# Negative lookahead filters the most common reference suffixes:
#   "Điều 5 của Luật..." / "Điều 5 này" / "Điều 5, khoản..."
# Note: references often lack a period directly after the number.
# ---------------------------------------------------------------------------
_DIEU_RE = re.compile(
    r'(?:Điều|ĐIỀU)\s+(\d+[a-zđ]?)\s*\.\s+',
    re.UNICODE,
)

# Words that follow a cross-reference "Điều X. <word>" — not an article title.
# "quy" was narrowed to "quy\s+định\s+<ref-word>" to avoid false-matching:
#   "Quyền" (rights), "Quyết định" (decision), "Quy trình" (procedure), etc.
#   The suffix list (này/đó/nêu/áp/đã/theo/như) identifies actual cross-reference
#   phrases ("quy định này", "quy định áp dụng") while passing article titles like
#   "Quy định chuyển tiếp" (Điều 218 LDN) and "Quy định chung".
_REF_NEXT = re.compile(
    r'^(?:của|này|đó|trên|sau|dưới|và|hoặc|nêu|nói|'
    r'quy\s+định\s+(?:này|đó|nêu|áp|đã|theo|như)|'
    r'khoản|điểm|,|;)',
    re.UNICODE | re.IGNORECASE,
)

# Khoản start inside an article: a 1-2 digit number + period + space at a
# "natural boundary" (preceded by whitespace or start-of-string).
_KHOAN_RE = re.compile(r'(?:^|\s+)(\d{1,2})\.\s+', re.UNICODE)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class Article:
    law_id: str
    law_type: str
    law_name: str          # without type prefix, e.g. "Doanh nghiệp"
    dieu_number: str       # "Điều 1", "Điều 24a"
    dieu_title: str        # may be empty for articles with no title
    content: str           # full body text of the article
    khoan_list: list       # [{"id": "1", "text": "..."}, ...]
    chunk_id: str          # md5 for dedup

    # Computed on init — included in asdict() for downstream use
    relevant_doc_str: str = field(init=False)
    relevant_article_str: str = field(init=False)

    def __post_init__(self):
        self.relevant_doc_str = f"{self.law_id}|{self.law_type} {self.law_name}"
        self.relevant_article_str = (
            f"{self.law_id}|{self.law_type} {self.law_name}|{self.dieu_number}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
class LegalTextParser:

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def parse(self, raw_text: str, law_meta: dict) -> list[Article]:
        """
        Parse raw text and return a list of Article objects.
        law_meta must have keys: id, type, name.
        """
        law_name = self._strip_type_prefix(law_meta["name"], law_meta["type"])
        text = re.sub(r"\s+", " ", raw_text).strip()

        boundaries = self._find_article_boundaries(text)
        if not boundaries:
            return []

        articles = []
        for i, (pos, num_str) in enumerate(boundaries):
            # Find where this article's content starts (after "Điều X. ")
            match_obj = _DIEU_RE.search(text, pos)
            if match_obj is None or match_obj.start() != pos:
                # Re-locate precisely
                match_obj = _DIEU_RE.search(text, pos, pos + 30)
            if match_obj is None:
                continue

            body_start = match_obj.end()
            body_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
            body = text[body_start:body_end].strip()
            if not body:
                continue

            dieu_number = f"Điều {num_str}"
            dieu_title, content = self._split_title_content(body)
            khoan_list = self._extract_khoan(content)

            chunk_id = md5(
                f"{law_meta['id']}|{dieu_number}|{content[:120]}".encode("utf-8")
            ).hexdigest()

            articles.append(
                Article(
                    law_id=law_meta["id"],
                    law_type=law_meta["type"],
                    law_name=law_name,
                    dieu_number=dieu_number,
                    dieu_title=dieu_title,
                    content=content,
                    khoan_list=khoan_list,
                    chunk_id=chunk_id,
                )
            )

        return articles

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _find_article_boundaries(self, text: str) -> list[tuple[int, str]]:
        """
        Return list of (position, num_str) for article-starting Điều markers.

        Two-stage filter:
        1. Negative lookahead: skip matches where the text immediately after
           the period is a reference indicator word.
        2. Monotonic sequence: articles are numbered 1,2,3... (with optional
           letter suffixes for amendments). Backward jumps are references.
        """
        candidates = []
        for m in _DIEU_RE.finditer(text):
            after = text[m.end() : m.end() + 30]
            if _REF_NEXT.match(after):
                continue
            candidates.append((m.start(), m.group(1)))  # (pos, num_str)

        # Monotonic filter
        result = []
        last_num = 0
        last_letter = ""

        for pos, num_str in candidates:
            digit_part = re.match(r"\d+", num_str).group()
            letter_part = num_str[len(digit_part):]  # '' or 'a','b','đ'...
            num = int(digit_part)

            if num > last_num:
                result.append((pos, num_str))
                last_num = num
                last_letter = letter_part
            elif num == last_num and letter_part and letter_part > last_letter:
                # Same number, next letter suffix (24a → 24b)
                result.append((pos, num_str))
                last_letter = letter_part
            # else: backward reference — skip

        return result

    def _split_title_content(self, body: str) -> tuple[str, str]:
        """
        Separate article title from content.

        Title = text before the first khoản marker (digit + period + space).
        If no khoản: title = first phrase up to 180 chars (trimmed to whole word).
        Content = full body.
        """
        # Look for first khoản: isolated "1. " pattern
        m = _KHOAN_RE.search(body)
        if m and m.start() < 300:
            title = body[: m.start()].strip()
            return title, body.strip()

        # No khoản — take up to 180 chars as title hint
        cap = min(180, len(body))
        title = body[:cap].strip()
        if len(body) > cap:
            # Trim to last space to avoid cutting a word
            cut = title.rfind(" ")
            if cut > 40:
                title = title[:cut]
        return title, body.strip()

    def _extract_khoan(self, content: str) -> list[dict]:
        """Extract numbered khoản from article content."""
        splits = list(_KHOAN_RE.finditer(content))
        if not splits:
            return []

        khoan_list = []
        for i, m in enumerate(splits):
            khoan_id = m.group(1)
            t_start = m.end()
            t_end = splits[i + 1].start() if i + 1 < len(splits) else len(content)
            khoan_text = content[t_start:t_end].strip()
            if khoan_text:
                khoan_list.append({"id": khoan_id, "text": khoan_text})
        return khoan_list

    @staticmethod
    def _strip_type_prefix(name: str, law_type: str) -> str:
        """
        Remove type prefix from name to avoid 'Luật Luật Doanh nghiệp'.
        Case-insensitive; handles multi-word types like 'Bộ luật'.
        """
        prefix = law_type.strip()
        if name.lower().startswith(prefix.lower()):
            stripped = name[len(prefix):].strip()
            if stripped:
                return stripped
        return name.strip()
