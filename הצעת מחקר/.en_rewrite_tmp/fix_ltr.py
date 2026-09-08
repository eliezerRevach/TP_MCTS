from __future__ import annotations

import re
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DC_NS = "http://purl.org/dc/elements/1.1/"
NS = {"w": W_NS, "dc": DC_NS}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def has_hebrew(text: str) -> bool:
    return bool(re.search(r"[\u0590-\u05FF]", text))


def get_or_add(parent: etree._Element, tag: str, before: tuple[str, ...] = ()) -> etree._Element:
    node = parent.find(w(tag))
    if node is not None:
        return node
    node = etree.Element(w(tag))
    before_qnames = {w(name) for name in before}
    for index, child in enumerate(parent):
        if child.tag in before_qnames:
            parent.insert(index, node)
            return node
    parent.append(node)
    return node


def replace_in_order(parent: etree._Element, tag: str, before: tuple[str, ...] = ()) -> etree._Element:
    for old in list(parent.findall(w(tag))):
        parent.remove(old)
    node = etree.Element(w(tag))
    before_qnames = {w(name) for name in before}
    for index, child in enumerate(parent):
        if child.tag in before_qnames:
            parent.insert(index, node)
            return node
    parent.append(node)
    return node


def set_val(node: etree._Element, value: str) -> None:
    node.set(w("val"), value)


def ensure_ppr(paragraph: etree._Element) -> etree._Element:
    ppr = paragraph.find(w("pPr"))
    if ppr is None:
        ppr = etree.Element(w("pPr"))
        paragraph.insert(0, ppr)
    return ppr


def ensure_rpr(run: etree._Element) -> etree._Element:
    rpr = run.find(w("rPr"))
    if rpr is None:
        rpr = etree.Element(w("rPr"))
        run.insert(0, rpr)
    return rpr


def force_paragraph_direction(ppr: etree._Element, rtl: bool) -> None:
    bidi = replace_in_order(
        ppr,
        "bidi",
        before=(
            "adjustRightInd",
            "snapToGrid",
            "spacing",
            "ind",
            "contextualSpacing",
            "mirrorIndents",
            "suppressOverlap",
            "jc",
            "textDirection",
            "textAlignment",
            "outlineLvl",
            "rPr",
            "sectPr",
            "pPrChange",
        ),
    )
    set_val(bidi, "1" if rtl else "0")
    mirror = ppr.find(w("mirrorIndents"))
    if mirror is not None:
        ppr.remove(mirror)


def force_run_direction(rpr: etree._Element, rtl: bool) -> None:
    rtl_node = replace_in_order(
        rpr,
        "rtl",
        before=("cs", "em", "lang", "eastAsianLayout", "specVanish", "oMath", "rPrChange"),
    )
    set_val(rtl_node, "1" if rtl else "0")
    lang = replace_in_order(rpr, "lang", before=("eastAsianLayout", "specVanish", "oMath", "rPrChange"))
    language = "he-IL" if rtl else "en-US"
    lang.set(w("val"), language)
    lang.set(w("eastAsia"), language)
    lang.set(w("bidi"), language)


def patch_story_part(root: etree._Element) -> None:
    for paragraph in root.xpath(".//w:p", namespaces=NS):
        text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))
        paragraph_is_hebrew = has_hebrew(text)
        ppr = ensure_ppr(paragraph)
        force_paragraph_direction(ppr, paragraph_is_hebrew)

        jc = ppr.find(w("jc"))
        if jc is None:
            jc = get_or_add(
                ppr,
                "jc",
                before=("textDirection", "textAlignment", "outlineLvl", "rPr", "sectPr", "pPrChange"),
            )
            set_val(jc, "right" if paragraph_is_hebrew else "left")

        for run in paragraph.xpath(".//w:r", namespaces=NS):
            run_text = "".join(run.xpath(".//w:t/text()", namespaces=NS))
            run_is_hebrew = has_hebrew(run_text) or (paragraph_is_hebrew and not run_text)
            force_run_direction(ensure_rpr(run), run_is_hebrew)

    for tblpr in root.xpath(".//w:tblPr", namespaces=NS):
        bidi_visual = replace_in_order(
            tblpr,
            "bidiVisual",
            before=(
                "tblStyleRowBandSize",
                "tblStyleColBandSize",
                "tblW",
                "jc",
                "tblCellSpacing",
                "tblInd",
                "tblBorders",
                "shd",
                "tblLayout",
                "tblCellMar",
                "tblLook",
                "tblCaption",
                "tblDescription",
                "tblPrChange",
            ),
        )
        set_val(bidi_visual, "0")

    for tcpr in root.xpath(".//w:tcPr", namespaces=NS):
        direction = replace_in_order(
            tcpr,
            "textDirection",
            before=("tcFitText", "vAlign", "hideMark", "headers", "cellIns", "cellDel", "cellMerge", "tcPrChange"),
        )
        set_val(direction, "lrTb")

    for sectpr in root.xpath(".//w:sectPr", namespaces=NS):
        rtl_gutter = replace_in_order(sectpr, "rtlGutter", before=("docGrid", "printerSettings"))
        set_val(rtl_gutter, "0")


def ensure_style_child(style: etree._Element, tag: str) -> etree._Element:
    node = style.find(w(tag))
    if node is not None:
        return node
    node = etree.Element(w(tag))
    if tag == "pPr":
        before = {w("rPr"), w("tblPr"), w("trPr"), w("tcPr"), w("tblStylePr")}
    else:
        before = {w("tblPr"), w("trPr"), w("tcPr"), w("tblStylePr")}
    for index, child in enumerate(style):
        if child.tag in before:
            style.insert(index, node)
            return node
    style.append(node)
    return node


def patch_styles(root: etree._Element) -> None:
    doc_defaults = root.find(w("docDefaults"))
    if doc_defaults is None:
        doc_defaults = etree.Element(w("docDefaults"))
        root.insert(0, doc_defaults)

    rpr_default = doc_defaults.find(w("rPrDefault"))
    if rpr_default is None:
        rpr_default = etree.SubElement(doc_defaults, w("rPrDefault"))
    rpr = rpr_default.find(w("rPr"))
    if rpr is None:
        rpr = etree.SubElement(rpr_default, w("rPr"))
    force_run_direction(rpr, False)

    ppr_default = doc_defaults.find(w("pPrDefault"))
    if ppr_default is None:
        ppr_default = etree.SubElement(doc_defaults, w("pPrDefault"))
    ppr = ppr_default.find(w("pPr"))
    if ppr is None:
        ppr = etree.SubElement(ppr_default, w("pPr"))
    force_paragraph_direction(ppr, False)

    for style in root.xpath(".//w:style", namespaces=NS):
        style_type = style.get(w("type"))
        if style_type == "paragraph":
            force_paragraph_direction(ensure_style_child(style, "pPr"), False)
        if style_type in {"paragraph", "character"}:
            force_run_direction(ensure_style_child(style, "rPr"), False)
        for tblpr in style.xpath("./w:tblPr", namespaces=NS):
            node = replace_in_order(tblpr, "bidiVisual", before=("tblW", "jc", "tblInd", "tblBorders", "shd", "tblLayout", "tblLook"))
            set_val(node, "0")


def patch_numbering(root: etree._Element) -> None:
    for level in root.xpath(".//w:lvl", namespaces=NS):
        level_jc = get_or_add(level, "lvlJc", before=("pPr", "rPr", "lvlPicBulletId"))
        set_val(level_jc, "left")
        ppr = get_or_add(level, "pPr", before=("rPr", "lvlPicBulletId"))
        force_paragraph_direction(ppr, False)
        ind = ppr.find(w("ind"))
        if ind is not None and w("right") in ind.attrib:
            ind.attrib.pop(w("right"), None)


def patch_settings(root: etree._Element) -> None:
    theme_lang = root.find(w("themeFontLang"))
    if theme_lang is None:
        theme_lang = etree.SubElement(root, w("themeFontLang"))
    theme_lang.set(w("val"), "en-US")
    theme_lang.set(w("eastAsia"), "en-US")
    theme_lang.set(w("bidi"), "en-US")

    for node in list(root.findall(w("activeWritingStyle"))):
        language = node.get(w("lang"), "")
        if language.lower().startswith(("ar", "he", "fa", "ur")):
            root.remove(node)


def patch_core(root: etree._Element) -> None:
    language = root.find(f"{{{DC_NS}}}language")
    if language is None:
        language = etree.SubElement(root, f"{{{DC_NS}}}language")
    language.text = "en-US"


def patch_xml(name: str, data: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False, resolve_entities=False)
    root = etree.fromstring(data, parser)
    if name in {"word/styles.xml", "word/stylesWithEffects.xml"}:
        patch_styles(root)
    elif name == "word/numbering.xml":
        patch_numbering(root)
    elif name == "word/settings.xml":
        patch_settings(root)
    elif name == "docProps/core.xml":
        patch_core(root)
    elif re.fullmatch(r"word/(document|header\d+|footer\d+|footnotes|endnotes|comments)\.xml", name):
        patch_story_part(root)
    else:
        return data
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def audit(docx_path: Path) -> None:
    with zipfile.ZipFile(docx_path) as archive:
        for name in ("word/styles.xml", "word/stylesWithEffects.xml"):
            root = etree.fromstring(archive.read(name))
            for style in root.xpath(".//w:style[@w:type='paragraph']", namespaces=NS):
                bidi = style.find(f"{w('pPr')}/{w('bidi')}")
                assert bidi is not None and bidi.get(w("val")) == "0", f"Non-LTR style in {name}"
            lang = root.find(f"{w('docDefaults')}/{w('rPrDefault')}/{w('rPr')}/{w('lang')}")
            assert lang is not None and lang.get(w("bidi")) == "en-US", f"BiDi language remains in {name}"

        root = etree.fromstring(archive.read("word/document.xml"))
        for paragraph in root.xpath(".//w:p", namespaces=NS):
            text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))
            bidi = paragraph.find(f"{w('pPr')}/{w('bidi')}")
            expected = "1" if has_hebrew(text) else "0"
            assert bidi is not None and bidi.get(w("val")) == expected, f"Wrong paragraph direction: {text[:80]}"
        for table in root.xpath(".//w:tbl", namespaces=NS):
            visual = table.find(f"{w('tblPr')}/{w('bidiVisual')}")
            assert visual is not None and visual.get(w("val")) == "0", "RTL table remains"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: fix_ltr.py INPUT.docx OUTPUT.docx")
    source = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx", dir=destination.parent) as handle:
        temp_path = Path(handle.name)
    try:
        with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(temp_path, "w") as dst:
            for info in src.infolist():
                data = src.read(info.filename)
                if info.filename.endswith(".xml"):
                    data = patch_xml(info.filename, data)
                dst.writestr(info, data)
        temp_path.replace(destination)
        audit(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    print(destination.name)


if __name__ == "__main__":
    main()
