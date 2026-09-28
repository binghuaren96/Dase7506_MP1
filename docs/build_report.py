"""Render the self-contained MP1 Markdown report as a compact, page-counted PDF.

This script is a document build tool, not an inference dependency. Install
reportlab separately and run ``python docs/build_report.py`` from the repository
root. The source report remains the editable document.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak, LongTable, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, TableStyle,
)


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "FINAL_REPORT.md"
OUTPUT = HERE / "FINAL_REPORT.pdf"
FONT_DIR = Path("C:/Windows/Fonts")


def register_fonts():
    pdfmetrics.registerFont(TTFont("ReportArial", str(FONT_DIR / "arial.ttf")))
    pdfmetrics.registerFont(TTFont("ReportArialBold", str(FONT_DIR / "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("ReportArialItalic", str(FONT_DIR / "ariali.ttf")))
    pdfmetrics.registerFontFamily(
        "ReportArial", normal="ReportArial", bold="ReportArialBold",
        italic="ReportArialItalic", boldItalic="ReportArialBold",
    )
    pdfmetrics.registerFont(TTFont("ReportCourier", str(FONT_DIR / "CascadiaMono.ttf")))


def styles():
    blue = colors.HexColor("#17324f")
    gray = colors.HexColor("#455468")
    return {
        "title": ParagraphStyle("Title", fontName="ReportArialBold", fontSize=18,
                                leading=22, textColor=blue, spaceAfter=10),
        "h2": ParagraphStyle("H2", fontName="ReportArialBold", fontSize=11.2,
                             leading=14, textColor=blue, spaceBefore=11, spaceAfter=5),
        "h3": ParagraphStyle("H3", fontName="ReportArialBold", fontSize=9.4,
                             leading=12, textColor=blue, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("Body", fontName="ReportArial", fontSize=9,
                               leading=12.6, textColor=gray, spaceAfter=6),
        "bullet": ParagraphStyle("Bullet", fontName="ReportArial", fontSize=9,
                                 leading=12.6, textColor=gray, leftIndent=12,
                                 firstLineIndent=-9, spaceAfter=2),
        "table": ParagraphStyle("Table", fontName="ReportArial", fontSize=7.5,
                                leading=10.2, textColor=gray),
        "thead": ParagraphStyle("TableHead", fontName="ReportArialBold", fontSize=7.5,
                                leading=10.2, textColor=colors.white),
        "code": ParagraphStyle("Code", fontName="ReportCourier", fontSize=6.9,
                               leading=9.2, textColor=gray, leftIndent=5),
    }


def inline(markdown: str) -> str:
    # ReportLab Paragraph accepts a small XML vocabulary. Escape user text first.
    value = html.escape(markdown, quote=False)
    value = re.sub(r"\[([^]]+)\]\(([^)]+)\)",
                   lambda m: f'<link href="{html.escape(m.group(2), quote=True)}">'
                             f'<u>{m.group(1)}</u></link>', value)
    value = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"`([^`]+)`", r'<font name="ReportCourier">\1</font>', value)
    for phrase in ("official", "same final update", "presentations"):
        value = value.replace(f"*{phrase}*", f"<i>{phrase}</i>")
    return value


def make_table(lines: list[str], s: dict, usable_width: float):
    rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    if not rows:
        return Spacer(1, 1)
    columns = max(map(len, rows))
    rows = [row + [""] * (columns - len(row)) for row in rows]
    # Give text-heavy columns more room while keeping every table inside the page.
    weights = [max(20, min(40, max(len(row[i]) for row in rows))) for i in range(columns)]
    widths = [usable_width * weight / sum(weights) for weight in weights]
    rendered = [[Paragraph(inline(cell), s["thead"] if ri == 0 else s["table"])
                 for cell in row] for ri, row in enumerate(rows)]
    table = LongTable(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#274a6a")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f1f5f8")]),
        ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#d4dce3")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def parse(source: str, s: dict, width: float):
    story = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line.strip():
            index += 1
            continue
        if line.startswith("```"):
            block = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                block.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(block), s["code"], maxLineLength=90))
            story.append(Spacer(1, 5))
            index += 1
            continue
        if line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].startswith("|"):
                block.append(lines[index])
                index += 1
            story.append(make_table(block, s, width))
            story.append(Spacer(1, 7))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:].strip()), s["title"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(CondPageBreak(55))
            story.append(Paragraph(inline(line[3:].strip()), s["h2"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(CondPageBreak(40))
            story.append(Paragraph(inline(line[4:].strip()), s["h3"]))
            index += 1
            continue
        if re.match(r"^([-*]|\d+\.)\s+", line):
            mark, content = line.split(" ", 1)
            bullet = "&#8226;" if mark in ("-", "*") else html.escape(mark)
            story.append(Paragraph(f"{bullet}  {inline(content)}", s["bullet"]))
            index += 1
            continue
        if re.fullmatch(r"-{3,}", line):
            index += 1
            continue
        paragraph = [line.strip()]
        index += 1
        while (index < len(lines) and lines[index].strip()
               and not re.match(r"^(#|\||```|[-*] |\d+\. )", lines[index])):
            paragraph.append(lines[index].strip())
            index += 1
        story.append(Paragraph(inline(" ".join(paragraph)), s["body"]))
    return story


def draw_page(canvas, doc):
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d4dce3"))
    canvas.setLineWidth(.5)
    canvas.line(doc.leftMargin, height - 31, width - doc.rightMargin, height - 31)
    canvas.setFont("ReportArial", 7)
    canvas.setFillColor(colors.HexColor("#647386"))
    canvas.drawString(doc.leftMargin, height - 24, "DASE7506 MP1  |  Technical report")
    canvas.drawRightString(width - doc.rightMargin, 24, f"Page {doc.page}")
    canvas.restoreState()


def main():
    register_fonts()
    s = styles()
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=39, rightMargin=39, topMargin=44, bottomMargin=39,
        title="DASE7506 MP1: Small Language Model Challenge",
        author="DASE7506 MP1 student",
    )
    story = parse(SOURCE.read_text(encoding="utf-8"), s,
                  A4[0] - document.leftMargin - document.rightMargin)
    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    try:
        import fitz
    except ImportError:
        print(f"Built {OUTPUT}; page count not checked (PyMuPDF unavailable)")
    else:
        with fitz.open(OUTPUT) as pdf:
            print(f"Built {OUTPUT} ({len(pdf)} pages)")
            if len(pdf) > 10:
                raise SystemExit("The report exceeds the course's 10-page limit")


if __name__ == "__main__":
    main()
