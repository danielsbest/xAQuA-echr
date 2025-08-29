from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, cast, Any
import re

NORMAL_FONT_PT = 12.0
SIZE_REMOVAL_WARNING_RATIO = 0.30  # warn if >30% of paragraphs removed by size -> could mean the default font size in this document is not 12pt as usual
MIN_ALPHA_CHARS = 5 # minimum number of alphabetic characters in a paragraph to consider it for removal

CSS_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
BLOCK_RE = re.compile(r"([^{}]+)\{([^{}]+)\}")
DECL_RE = re.compile(r"(?P<prop>[-a-zA-Z]+)\s*:\s*(?P<val>[^;]+)")
CLASS_SELECTOR_RE = re.compile(r"\.([A-Za-z_][\w\-]*)")


def parse_style_attr(style_str: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not style_str:
        return out
    for m in DECL_RE.finditer(style_str):
        prop = m.group("prop").strip().lower()
        val = m.group("val").strip()
        out[prop] = val
    if "font" in out:
        font_val = out["font"]
        tokens = re.split(r"\s+|/", font_val.strip())
        for t in tokens:
            tl = t.lower()
            if tl in ("bold", "bolder", "lighter") or re.match(r"^[1-9]00$", tl):
                out.setdefault("font-weight", t)
            if tl.endswith("pt") or tl.endswith("px"):
                out.setdefault("font-size", t)
    return out


def _attr_to_str(val: Any) -> str:
    """Normalize a BeautifulSoup attribute value to a string."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x is not None)
    return str(val)


def _get_classes(tag: Tag) -> List[str]:
    """Return classes as a list of strings, robust to bs4 types."""
    raw = tag.get("class")
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None]
    return [str(raw)]


def extract_css_class_map(soup: BeautifulSoup) -> Dict[str, Dict[str, str]]:
    css_map: Dict[str, Dict[str, str]] = defaultdict(dict)
    for st in soup.find_all("style"):
        st_tag = cast(Tag, st)
        css_text = str(st_tag.string or "")
        if not css_text:
            continue
        css_text = CSS_COMMENT_RE.sub("", css_text)
        for m in BLOCK_RE.finditer(css_text):
            selectors = m.group(1)
            body = m.group(2)
            decls = parse_style_attr(body)
            for sel in selectors.split(","):
                sel = sel.strip()
                for cm in CLASS_SELECTOR_RE.finditer(sel):
                    cls = cm.group(1)
                    for key in (
                        "font-weight",
                        "font-size",
                        "text-transform",
                        "font-variant",
                        "font-variant-caps",
                        "font",
                    ):
                        if key in decls:
                            css_map[cls][key] = decls[key]
    return css_map


def parse_font_size_to_pt(val: Optional[str]) -> Optional[float]:
    if not val:
        return None
    v = val.strip().lower()
    m = re.search(r"([0-9]*\.?[0-9]+)\s*(pt|px)", v)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2)
    if unit == "pt":
        return num
    if unit == "px":
        # 1px = 0.75pt at 96dpi
        return num * 0.75
    return None

def make_style_getter(css_map: Dict[str, Dict[str, str]]):
    cache: Dict[int, Dict[str, str]] = {}

    def get_effective_props(el: Tag) -> Dict[str, str]:
        key = id(el)
        if key in cache:
            return cache[key]

        effective: Dict[str, str] = {}
        cur: Optional[Tag] = el

        def merge(d: Dict[str, str]):
            for k, v in d.items():
                if k not in effective and v:
                    effective[k] = v

        while isinstance(cur, Tag):
            # b/strong imply bold if not already set
            if cur.name in ("b", "strong") and "font-weight" not in effective:
                effective["font-weight"] = "bold"

            merge(parse_style_attr(_attr_to_str(cur.get("style"))))

            for cls in _get_classes(cur):
                if cls in css_map:
                    merge(css_map[cls])

            if cur.name in ("body", "html"):
                break
            cur = cur.parent  # type: ignore

        cache[key] = effective
        return effective

    return get_effective_props


def is_bold_value(val: Optional[str]) -> bool:
    if not val:
        return False
    v = val.strip().lower()
    if v == "bold":
        return True
    m = re.match(r"^([1-9]00)$", v)
    if m:
        return int(m.group(1)) >= 600
    return False


def has_uppercase_style(props: Dict[str, str]) -> bool:
    tt = (props.get("text-transform") or "").strip().lower()
    if tt == "uppercase":
        return True
    fv = (props.get("font-variant") or "").strip().lower()
    fvc = (props.get("font-variant-caps") or "").strip().lower()
    return "small-caps" in fv or "small-caps" in fvc

def iter_text_nodes(el: Tag):
    for node in el.descendants:
        if isinstance(node, NavigableString):
            yield node


def count_alpha(s: str) -> int:
    return sum(1 for ch in s if ch.isalpha())


def has_lower_alpha(s: str) -> bool:
    return any(ch.isalpha() and ch.islower() for ch in s)

HEADING_CLASS_PATTERNS = [
    re.compile(r"\bmsoheading[0-9]+\b", re.I),
    re.compile(r"\bheading[0-9]+\b", re.I),
    re.compile(r"\bmso(title|subtitle)\b", re.I),
    re.compile(r"\btitle\b", re.I),
    re.compile(r"\bsubtitle\b", re.I),
]

MSO_STYLE_NAME_RE = re.compile(
    r"mso-style-name\s*:\s*['\"]?\s*(Heading\s*\d+|Title|Subtitle)\b",
    re.I,
)


def is_conventional_heading_tag(tag: Tag) -> bool:
    if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return True
    role = _attr_to_str(tag.get("role"))
    if role.lower() == "heading":
        return True

    classes = " ".join(_get_classes(tag)).lower()
    if classes:
        for pat in HEADING_CLASS_PATTERNS:
            if pat.search(classes):
                return True

    style = _attr_to_str(tag.get("style"))
    if style and MSO_STYLE_NAME_RE.search(style):
        return True
    return False


def paragraph_is_all_caps(
    p: Tag,
    get_props,
    normal_pt: float
) -> bool:
    total_letters = 0
    for tn in iter_text_nodes(p):
        txt = str(tn)
        if not txt.strip():
            continue

        props = get_props(tn.parent if isinstance(tn.parent, Tag) else p)
        size_pt = None
        if props.get("font-size"):
            size_pt = parse_font_size_to_pt(props.get("font-size"))
        if size_pt is None and props.get("font"):
            size_pt = parse_font_size_to_pt(props.get("font"))

        size_ok = size_pt is None or (size_pt is not None and size_pt >= normal_pt)

        if has_uppercase_style(props):
            if not size_ok and count_alpha(txt) > 0:
                return False
            total_letters += count_alpha(txt) if size_ok else 0
            continue

        for ch in txt:
            if ch.isalpha():
                if not size_ok:
                    return False
                total_letters += 1
                if ch.islower():
                    return False

    return total_letters >= MIN_ALPHA_CHARS


def paragraph_is_all_bold(
    p: Tag,
    get_props,
    normal_pt: float
) -> bool:
    total_letters = 0

    # Paragraph-level bold allows quick acceptance if there is text
    p_props = get_props(p)
    para_bold = is_bold_value(p_props.get("font-weight"))

    for tn in iter_text_nodes(p):
        txt = str(tn)
        run_props = get_props(tn.parent if isinstance(tn.parent, Tag) else p)
        size_pt = None
        if run_props.get("font-size"):
            size_pt = parse_font_size_to_pt(run_props.get("font-size"))
        if size_pt is None and run_props.get("font"):
            size_pt = parse_font_size_to_pt(run_props.get("font"))
        size_ok = size_pt is None or (size_pt is not None and size_pt >= normal_pt)

        for ch in txt:
            if not ch.isalpha():
                continue
            total_letters += 1
            if not size_ok:
                return False
            if para_bold:
                continue
            if not is_bold_value(run_props.get("font-weight")):
                return False

    return total_letters >= MIN_ALPHA_CHARS


def paragraph_all_runs_larger_than(
    p: Tag,
    get_props,
    threshold_pt: float
) -> bool:
    saw_any = False
    for tn in iter_text_nodes(p):
        txt = str(tn)
        if not txt.strip():
            continue
        saw_any = True
        props = get_props(tn.parent if isinstance(tn.parent, Tag) else p)
        size_pt = None
        if props.get("font-size"):
            size_pt = parse_font_size_to_pt(props.get("font-size"))
        if size_pt is None and props.get("font"):
            size_pt = parse_font_size_to_pt(props.get("font"))
        if size_pt is None or size_pt <= threshold_pt:
            return False
    return saw_any

def remove_headlines_simple(
    html: str,
    normal_pt: float = NORMAL_FONT_PT,
    size_warning_ratio: float = SIZE_REMOVAL_WARNING_RATIO,
) -> Tuple[str, Dict]:
    soup = BeautifulSoup(html, "lxml")
    css_map = extract_css_class_map(soup)
    get_props = make_style_getter(css_map)

    report = {
        "removed_counts": {
            "conventional_headings": 0,
            "all_caps": 0,
            "all_bold": 0,
            "size": 0,
        },
        "total_paragraphs": 0,
        "size_removal_ratio": 0.0,
        "warnings": [],
    }

    for tag in list(soup.find_all(True)):
        if is_conventional_heading_tag(cast(Tag, tag)):
            report["removed_counts"]["conventional_headings"] += 1
            cast(Tag, tag).decompose()

    paragraphs = list(soup.find_all("p"))
    report["total_paragraphs"] = len(paragraphs)

    for p in paragraphs:
        if not p.get_text(strip=True):
            # ignore empty paragraphs (don't count as removed)
            continue

        if paragraph_is_all_caps(cast(Tag, p), get_props, normal_pt):
            report["removed_counts"]["all_caps"] += 1
            cast(Tag, p).decompose()
            continue

        if paragraph_is_all_bold(cast(Tag, p), get_props, normal_pt):
            report["removed_counts"]["all_bold"] += 1
            cast(Tag, p).decompose()
            continue

        if paragraph_all_runs_larger_than(cast(Tag, p), get_props, normal_pt):
            report["removed_counts"]["size"] += 1
            cast(Tag, p).decompose()
            continue

    total_p = max(1, report["total_paragraphs"])
    size_ratio = report["removed_counts"]["size"] / total_p
    report["size_removal_ratio"] = size_ratio
    if size_ratio > size_warning_ratio:
        report["warnings"].append(
            f"Size-based filter removed {size_ratio:.0%} of paragraphs. "
            "Assumed normal size may not be 12pt."
        )

    return str(soup), report

def available_paragraphs(text: str) -> int:
    """Count how many numbered paragraphs exist in the text."""
    i = 1
    while f"\n{i}" in text:
        i += 1
    return i - 1


def parse_judgement_paragraphs(html_content: str, strip_headlines: bool = True) -> Dict[int, str]:
    """Extract numbered paragraphs from raw HUDOC judgement HTML.

    Steps:
    1. Optionally remove headline-like paragraphs (conventional headings, ALL CAPS, ALL BOLD, or oversized) using
       `remove_headlines_simple` to avoid polluting numbered paragraph extraction.
    2. Plain-text extraction via BeautifulSoup with newlines as separators.
    3. Heuristic enumeration: paragraphs are assumed to be introduced by a newline, then an integer and a dot, e.g. '\n123.'

    Args:
        html_content: Raw HTML of the judgement body.
        strip_headlines: If True (default), run `remove_headlines_simple` first. Only the cleaned HTML is used if the
                        cleaning report contains no warnings.

    Returns:
        Dict mapping paragraph number -> normalized paragraph text (prefixed with its number and a dot).
    """
    if strip_headlines:
        try:
            cleaned_html, report = remove_headlines_simple(html_content)
            if not report.get("warnings"):
                html_content = cleaned_html
            else:
                print("Warnings found in headlines cleaning report, kept headlines:")
                for warning in report.get("warnings", []):
                    print(f" - {warning}")
        except Exception:
            print("Failed to clean headlines from HTML content.")
            pass

    soup = BeautifulSoup(html_content, "html.parser")
    text_content = soup.get_text(separator="\n")
    n = available_paragraphs(text_content)

    def _find_paragraph_marker(search_text: str, idx: int) -> Optional[re.Match]:
        """Return first match object for paragraph number idx using hierarchical patterns.

        Pattern priority:
        1. Newline + number + optional space + dot
        2. Number + optional space + dot, not preceded by digit or '/'
        3. Bare number (no dot), not flanked by '/', '.' (nor preceded by digit)
        4. Newline + number with duplicated last digit + dot (OCR duplication)
        """
        last_digit_local = str(idx)[-1]
        patterns = [
            rf"\n({idx})\s?\.",
            rf"(?<![\d/])({idx})\s?\.",
            rf"(?<![\d/\.])({idx})(?![\/\.])",
            rf"\n({idx}{last_digit_local})\.",
        ]
        for pat in patterns:
            m = re.search(pat, search_text)
            if m:
                return m
        return None

    paragraphs: Dict[int, str] = {}
    text = text_content
    for i in range(1, n):
        m_curr = _find_paragraph_marker(text, i)
        m_next = _find_paragraph_marker(text, i + 1)

        if not m_curr:
            continue

        start_curr = m_curr.end()
        if m_next and m_next.start() >= start_curr:
            paragraph = text[start_curr:m_next.start()]
            text = text[m_next.start():]
        else:
            paragraph = text[start_curr:]
            text = ""

        paragraph_text = re.sub(r'\s+', ' ', paragraph).strip()
        if paragraph_text:  # Only include non-empty paragraphs
            paragraphs[i] = f"{i}. {paragraph_text}"
    if n > 0:
        m_n = _find_paragraph_marker(text, n)
        if m_n:
            paragraph = text[m_n.end():]
            paragraph_text = re.sub(r'\s+', ' ', paragraph).strip()
            if paragraph_text:
                paragraphs[n] = f"{n}. {paragraph_text}"
        # If no marker or empty content, omit the final paragraph.
        print("Successfully extracted paragraphs")
    return paragraphs