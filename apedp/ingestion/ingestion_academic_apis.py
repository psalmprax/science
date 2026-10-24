"""APEDP academic/institutional API target configuration."""


INSTITUTIONAL_TARGETS: dict[str, str] = {
    "arXiv": "http://export.arxiv.org/api/query",
    "HAL Open Science": "https://api.archives-ouvertes.fr/search",
    "CERN Document Server": "https://cds.cern.ch/search",
    "NASA ADS": "https://api.adsabs.harvard.edu/v1/search",
    "NIST DLMF": "https://dlmf.nist.gov/",
    "Materials Project (LBNL)": "https://api.materialsproject.org/",
    "PTB Germany": "https://www.ptb.de/",
    "Wolfram Alpha": "https://api.wolframalpha.com/v2/query",
    "Wikidata": "https://query.wikidata.org/sparql",
    "zbMATH Open": "https://api.zbmath.org/v2/",
    "CODATA": "https://codata.org/",
    "PubChem": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/",
    "MIT OCW": "https://ocw.mit.edu/",
    "Caltech JPL": "https://naif.jpl.nasa.gov/",
    "Princeton IAS": "https://www.ias.edu/",
    "Max Planck Society": "https://www.mpg.de/",
    "ETH Zurich": "https://ethz.ch/",
    "Stanford Encyclopedia": "https://plato.stanford.edu/",
    "Imperial College London": "https://www.imperial.ac.uk/",
    "University of Tokyo": "https://www.u-tokyo.ac.jp/",
}


def list_targets() -> dict[str, str]:
    """Return configured institutional targets."""
    return dict(INSTITUTIONAL_TARGETS)
