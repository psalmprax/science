"""Wikipedia equation harvesting module for APEDP."""
import json
import logging
import re
import ssl
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

USER_AGENT = 'SciencePlatformAssurance/1.0 (contact@scienceplatform.org)'


def _fetch_page_html(page_title: str) -> str:
    try:
        encoded_title = urllib.parse.quote(page_title)
        url = f"https://en.wikipedia.org/w/api.php?action=parse&page={encoded_title}&prop=text&format=json"
        ctx = ssl.create_default_context()
        # URL scheme/host are hardcoded constants (https://en.wikipedia.org);
        # only the page title (URL-quoted above) varies. The noqa sits on this
        # line because ruff anchors S310 to the Request/urlopen pair and the
        # previous placement was one line below the reported column, so the
        # suppression never applied and the finding had been sitting in CI as
        # a permanently-red step.
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})  # noqa: S310
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:  # noqa: S310  # nosec B310 - hardcoded https host
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
    latex_candidates = re.findall(r'{\displaystyle\s*([^"}]+)}', html_content)
    alt_candidates = re.findall(r'alt="([^"]+)"', html_content)
    combined = list(set(latex_candidates + alt_candidates))

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
