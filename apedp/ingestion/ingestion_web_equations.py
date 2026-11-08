"""Wikipedia equation harvesting module for APEDP."""
import json
import logging
import re
import ssl
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

USER_AGENT = 'SciencePlatformAssurance/1.0 (contact@scienceplatform.org)'

#: The ``displaystyle`` scan is brace-balanced, so a nested group survives whole,
#: but it is still lossy: macros are not expanded, math outside a ``displaystyle``
#: span is skipped, and ``alt=`` text is taken without verification. Every
#: harvested string must be re-verified before it is treated as an equation, as
#: ``docs/SCAFFOLD_INVENTORY.md`` records.
EXTRACTION_IS_LOSSY = True

#: The literal opening of a MediaWiki math span. It contains a backslash, so a
#: regex needs a doubled escape; a plain ``str.find`` does not, and the scan below
#: then only has to count braces.
_DISPLAYSTYLE_OPEN = '{\\displaystyle'


def _displaystyle_bodies(html_content: str) -> list[str]:
    """Bodies of every ``{\\displaystyle ...}`` span, brace-balanced.

    The old pattern was a flat ``[^"}]`` class, which stopped at the first inner
    brace, so ``x^{2}`` came back as ``x^{2``. Counting depth to the matching
    close returns the group whole. An escaped brace (``\\{``, ``\\}``) is a
    literal character, not a group boundary, and is skipped.
    """
    bodies: list[str] = []
    search_from = 0
    while True:
        opened = html_content.find(_DISPLAYSTYLE_OPEN, search_from)
        if opened < 0:
            return bodies
        position = opened + len(_DISPLAYSTYLE_OPEN)
        begin = position
        depth = 1
        while position < len(html_content) and depth:
            character = html_content[position]
            if character == '\\':
                position += 2  # `\{` / `\}` are literal, and macros are opaque here
                continue
            if character == '{':
                depth += 1
            elif character == '}':
                depth -= 1
                if depth == 0:
                    break
            position += 1
        if depth:
            return bodies  # unterminated span: nothing after it can be trusted
        bodies.append(html_content[begin:position])
        search_from = position + 1


def _fetch_page_html(page_title: str) -> str:
    try:
        encoded_title = urllib.parse.quote(page_title)
        url = f"https://en.wikipedia.org/w/api.php?action=parse&page={encoded_title}&prop=text&format=json"
        ctx = ssl.create_default_context()
        # URL scheme/host are hardcoded constants (https://en.wikipedia.org);
        # only the page title (URL-quoted above) varies. Ruff reports S310 once
        # for `Request` and once for `urlopen`, while bandit reports B310 on
        # `urlopen` alone, so both call sites carry a noqa and the urlopen line
        # also carries the nosec reason. Suppressing only one left the other
        # finding permanently red.
        req = urllib.request.Request(  # noqa: S310
            url, headers={'User-Agent': USER_AGENT}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:  # noqa: S310  # nosec B310
            data = json.loads(response.read().decode('utf-8'))
        parsed = data.get('parse', {}) if isinstance(data, dict) else {}
        text = parsed.get('text', {}) if isinstance(parsed, dict) else {}
        formula = text.get('*', '') if isinstance(text, dict) else ''
        return formula if isinstance(formula, str) else ''
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to fetch Wikipedia page '%s': %s", page_title, e)
        return ''


def _extract_formula_strings(html_content: str) -> list[str]:
    """Extract LaTeX/alt text candidates from Wikipedia HTML."""
    latex_candidates = [body.strip() for body in _displaystyle_bodies(html_content)]
    alt_candidates = re.findall(r'alt="([^"]+)"', html_content)
    combined = list(dict.fromkeys(latex_candidates + alt_candidates))

    formulas: list[str] = []
    for f in combined:
        f_clean = f.strip().replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        if (
            '=' in f_clean
            and len(f_clean) > 3
            and not any(noise in f_clean for noise in ['\\begin', '\\align', 'class=', 'width='])
        ):
            formulas.append(f_clean)
    return sorted(set(formulas))


def harvest_wikipedia_equations(page_title: str) -> list[str]:
    """Return a list of raw equation strings from a Wikipedia page."""
    html = _fetch_page_html(page_title)
    return _extract_formula_strings(html)


DISCIPLINE_PAGES: dict[str, str] = {
    'Classical Mechanics': 'List of equations in classical mechanics',
    'Electromagnetism': 'List of electromagnetism equations',
    'Fluid Mechanics': 'List of equations in fluid mechanics',
    'Gravitation': 'List of equations in gravitation',
    'Physical Chemistry': 'Defining equation (physical chemistry)',
}


def harvest_all_disciplines() -> dict[str, list[str]]:
    """Harvest equations for all configured disciplines."""
    library: dict[str, list[str]] = {}
    for discipline, page in DISCIPLINE_PAGES.items():
        try:
            formulas = harvest_wikipedia_equations(page)
            library[discipline] = formulas
        except (OSError, ValueError, TimeoutError) as e:
            logger.warning("Harvest failed for discipline '%s' (%s)", discipline, e)
            library[discipline] = []
    return library
