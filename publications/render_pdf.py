"""Render publications/manuscript.md to PDF via reportlab (DejaVu Unicode).

Usage: .venv/bin/python publications/render_pdf.py
Output: publications/H-A_volatility_forecasting.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "manuscript.md"
PDF = HERE / "H-A_volatility_forecasting.pdf"

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

LATEX_CACHE = HERE / ".mathcache"
LATEX_CACHE.mkdir(exist_ok=True)


def render_math(tex: str, fontsize: int = 15) -> str:
    """Render LaTeX math via matplotlib mathtext to PNG; return path."""
    import hashlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # mathtext-safe preprocessing (no \boxed in mathtext: unwrap, keep content)
    t = tex.strip()
    t = re.sub(r"\\boxed\s*\{", "{", t)
    t = re.sub(r"\\text\{([^}]*)\}", r"\\mathrm{\1}", t)
    key = hashlib.md5(f"{fontsize}|{t}".encode()).hexdigest()
    out = LATEX_CACHE / f"{key}.png"
    if not out.exists():
        # Pixel-ink crop: bbox_inches='tight' silently returns the full
        # figure here (the invisible axes patch poisons the extent union).
        from PIL import Image as _PILImage
        fig = plt.figure(figsize=(6.5, 1.0), dpi=200)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"${t}$", fontsize=fontsize, ha="center",
                va="center", transform=ax.transAxes)
        tmp = str(out) + ".full.png"
        fig.savefig(tmp, dpi=200, facecolor="white")
        plt.close(fig)
        im = _PILImage.open(tmp).convert("RGB")
        import numpy as _np
        a = _np.asarray(im)
        ys, xs = _np.where(a.sum(axis=2) < 720)
        if len(xs) == 0:
            raise ValueError(f"empty math render: {t[:40]}")
        pad = 12
        im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                 min(im.width, xs.max() + pad), min(im.height, ys.max() + pad))).save(out)
        try:
            import os as _os
            _os.remove(tmp)
        except OSError:
            pass
    return str(out)


def img_tag(path: str, height_pt: float = 11.0) -> str:
    """ReportLab <img> sized to paragraph line height (aspect preserved)."""
    from PIL import Image as _PILImage
    w, h = _PILImage.open(path).size
    return (f'<img src="{path}" width="{w / h * height_pt:.1f}" '
            f'height="{height_pt:.1f}" valign="-3"/>')

for fam, fn in (("DejaVu", "DejaVuSans.ttf"),
                ("DejaVu-Bold", "DejaVuSans-Bold.ttf"),
                ("DejaVu-Oblique", "DejaVuSans.ttf")):
    pdfmetrics.registerFont(TTFont(fam, f"/usr/share/fonts/truetype/dejavu/{fn}"))

TEX = [
    (r"\\boxed\s*\{", "["), (r"\\boxed\{", "["),
    (r"\\text\{([^}]*)\}", r"\1"), (r"\\mathrm\{([^}]*)\}", r"\1"),
    (r"\\frac\{([^}]*)\}\{([^}]*)\}", r"(\1)/(\2)"),
    (r"\\sum_\{([^}]*)\}", r"sum_\1"), (r"\\sum", "sum"),
    (r"\\rightarrow", "\u2192"), (r"\\Rightarrow", "\u21d2"),
    (r"\\neq", "\u2260"), (r"\\approx", "\u2248"), (r"\\leq", "\u2264"),
    (r"\\geq", "\u2265"), (r"\\times", "\u00d7"), (r"\\pm", "\u00b1"),
    (r"\\in", "\u2208"), (r"\\ldots", "..."), (r"\\cdots", "..."),
    (r"\\epsilon", "\u03b5"), (r"\\beta", "\u03b2"), (r"\\sigma", "\u03c3"),
    (r"\\alpha", "\u03b1"), (r"\\rho", "\u03c1"), (r"\\tau", "\u03c4"),
    (r"\\Delta", "\u0394"), (r"\\Omega", "\u03a9"), (r"\\hat\{([^}]*)\}", r"\1^"),
    (r"\\bar ([A-Za-z])", r"\1-bar"), (r"\\quad", " "), (r"\\qquad", " "),
    (r"\\_", "_"), (r"\\%", "%"), (r"\\&", "&"),
]
MD_TO_TAG = [
    (r"\*\*(.+?)\*\*", r"<b>\1</b>"),
    (r"`([^`]+)`", r'<font face="DejaVu-Oblique">\1</font>'),
]


def clean(s: str) -> str:
    for pat, rep in TEX:
        s = re.sub(pat, rep, s)
    s = s.replace("{", "").replace("}", "")
    for pat, rep in MD_TO_TAG:
        s = re.sub(pat, rep, s)
    return s.strip()


ST = {
    "title": ParagraphStyle("title", fontName="DejaVu-Bold", fontSize=19, leading=24, alignment=1, spaceAfter=6),
    "h1": ParagraphStyle("h1", fontName="DejaVu-Bold", fontSize=14, leading=18, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", fontName="DejaVu-Bold", fontSize=12, leading=16, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="DejaVu", fontSize=11, leading=15, spaceAfter=5, alignment=4),
    "math": ParagraphStyle("math", fontName="DejaVu", fontSize=11, leading=15, spaceAfter=5, alignment=1,
                           textColor="#0b3d5c"),
    "cell": ParagraphStyle("cell", fontName="DejaVu", fontSize=9, leading=12),
    "cellh": ParagraphStyle("cellh", fontName="DejaVu-Bold", fontSize=9, leading=12),
}


def cell_txt(s: str, hdr: bool = False) -> Paragraph:
    return Paragraph(clean(s), ST["cellh"] if hdr else ST["cell"])


def para_with_math(s: str) -> str:
    r"""Paragraph text with inline \(...\) spans rendered as LaTeX images."""
    spans: list[str] = []

    def stash(m: "re.Match[str]") -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x01"

    s = re.sub(r"\\\((.+?)\\\)", stash, s)
    s = clean(s)
    for i, tex in enumerate(spans):
        try:
            s = s.replace(f"\x00{i}\x01", img_tag(render_math(tex, fontsize=12), height_pt=12.0))
        except Exception:
            s = s.replace(f"\x00{i}\x01", clean(tex))
    return s


def build() -> None:
    lines = MD.read_text().split("\n")
    story = []
    i = 0
    # Title = first # line
    while i < len(lines):
        if lines[i].startswith("# "):
            story.append(Paragraph(clean(lines[i][2:]), ST["title"]))
            i += 1
            break
        i += 1
    in_math = False
    mathbuf: list[str] = []
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("$$"):
            if not in_math:
                in_math = True
                mathbuf = []
                rest = s[2:].strip()
                if rest:
                    mathbuf.append(rest)
            else:
                in_math = False
                try:
                    story.append(Image(render_math(" ".join(mathbuf)), width=14 * cm,
                                       height=1.8 * cm, kind="proportional"))
                except Exception as exc:  # fall back to text on render failure
                    story.append(Paragraph(clean(" ".join(mathbuf)), ST["math"]))
            i += 1
            continue
        if in_math:
            mathbuf.append(s)
            i += 1
            continue
        if not s:
            i += 1
            continue
        if s.startswith("## "):
            story.append(Paragraph(clean(s[3:]), ST["h1"]))
        elif s.startswith("### "):
            story.append(Paragraph(clean(s[4:]), ST["h2"]))
        elif s.startswith("# "):
            story.append(Paragraph(clean(s[2:]), ST["h1"]))
        elif s.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|\s*$", lines[i + 1].strip()):
            hdr = [c.strip() for c in s.strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ncol = len(hdr)
            data = [[cell_txt(c, True) for c in hdr]]
            for r in rows:
                r = (r + [""] * ncol)[:ncol]
                data.append([cell_txt(c) for c in r])
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.4, "#999999"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
            continue
        elif re.match(r"^(\d+\.|[-*])\s", s):
            story.append(Paragraph(para_with_math(re.sub(r"^(\d+\.|[-*])\s+", "• ", s)), ST["body"]))
        else:
            story.append(Paragraph(para_with_math(s), ST["body"]))
        i += 1
    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm,
                            title="A Falsification-Oriented Adaptive Framework for Log-Volume Volatility Forecasting")
    doc.build(story)
    print(f"wrote {PDF} ({PDF.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build()
