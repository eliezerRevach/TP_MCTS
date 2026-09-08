# Template execution contract

## Reference

- Retained DOCX: `C:\Users\eliezer\Documents\hw2_exam\TP_MCTS\הצעת מחקר\הצעת_מחקר_PDB_אליעזר_רווח.docx`
- SHA-256: `DAA0C3BD434E9367880FC484EFF3508FA15DF8E2ACEF66ACB8AF39C8F8FACF46`
- Reference render: `C:\Users\eliezer\Documents\hw2_exam\TP_MCTS\הצעת מחקר\qa_old`
- Evidence: `template-style-evidence.json`, section/style/heading/field/image/footnote/content-control audits.
- Page count: 4. Section count: 1.

## Page system

- A4 portrait, 8.2681 x 11.6931 in.
- Margins: left/right 0.6493 in, top 0.5708 in, bottom 0.5312 in.
- Header/footer distance: 0.2562 in. Header and footer contain no visible text.
- One section; no different-first-page or odd/even-page variants.
- Explicit page-break paragraphs are top-level body paragraphs 4, 13, 27, and 41. They are preserve-only and define the four-page structure.

## Typography and recurring roles

- Base family: Arial; body ink `#17202A`.
- Normal: 9.5 pt, 1.0792 line spacing, 3.2 pt after; justified for narrative body text.
- Heading 1: Arial 13 pt bold, `#1F4E79`, 7 pt before, 3 pt after, single spacing, keep-with-next.
- Heading 2: Arial 10.5 pt bold, `#17365D`, 4.5 pt before, 2 pt after, single spacing, keep-with-next.
- University line: centered Arial 9 pt bold, `#5B6573`, 1 pt after.
- Proposal label: centered Arial 11.5 pt bold, `#1F4E79`, 3 pt after.
- Hebrew title: centered Arial 16.5 pt bold, `#17365D`, single spacing, 2 pt after, RTL.
- English title: centered Arial 9.5 pt italic, `#5B6573`, 5 pt after, LTR.
- Display equation: centered Arial 10 pt bold italic, `#17365D`, 2 pt before and 4 pt after.
- References: Arial 7.5 pt, left aligned, single spacing, 1 pt after, real decimal numbering inherited from the source.
- Metadata and note runs: Arial 8.5 pt; labels bold, values regular.

## Lists and tables

- Research questions use the source's real bullet numbering and paragraph indents; change only text and direction.
- References use the source's real decimal numbering; preserve numbering definitions and indents.
- Metadata table: 3 x 2, fixed width 10036 DXA, grid 5018/5018, centered, zero table indent, pale blue borders and source cell shading/margins.
- Decision note: 1 x 1, fixed width 10036 DXA, centered, zero table indent, gold border and pale-yellow cell shading.
- Do not add rows, fixed row heights, or new tables.

## Content flow and slot map

- Top-level body p0-p3: university, proposal label, Hebrew title, English title. Rewrite in place; retain centered treatment. Only p2 remains RTL.
- Table 0 cells r0c0-r2c1: personal metadata. Rewrite in English. Use `Program: Computer Science`; retain `[TO COMPLETE]` for student ID and email.
- p5-p12: Abstract and Scientific Background, page 1. Rewrite in academic English.
- p14-p26: Research Objective and Questions, Formal Setting and Proof Objectives, Proposed Novelty, page 2. Rewrite in academic English.
- p28-p40: Research Methodology and Optional Extension, page 3. Rewrite in academic English.
- p42-p47: Evaluation Plan, Expected Contributions, References heading, page 4. Rewrite in academic English.
- p48-p55: English references. Preserve bibliographic content; remove RTL paragraph/run direction only.
- Table 1 r0c0: unresolved decisions note. Rewrite in English and preserve the callout treatment.
- p4, p13, p27, p41: preserve exactly as page-break carriers.

## Package preservation

Only `word/document.xml` and the source-derived bullet definition 91 in `word/numbering.xml` are editable. The latter must be mirrored from right-indented RTL geometry to equivalent left-indented LTR geometry while retaining the real bullet definition. Every other part and relationship is preserve-only and must retain its exact uncompressed bytes and SHA-256 digest.

| Part | Bytes | SHA-256 |
|---|---:|---|
| `[Content_Types].xml` | 2122 | `43520c56fa8b7a4023384034b373ba13e257e3c35258a134c5b4ec72dfb2ba39` |
| `_rels/.rels` | 734 | `a9ae57efe9186f07d48303bcac5d54c7e359bf3f503939c03ad2954e5c59a5c4` |
| `docProps/core.xml` | 967 | `d6420c7f329ef1f1d3c654e55ee2a4d96ef516084d3b1680bbc5eee1109035df` |
| `docProps/app.xml` | 1132 | `be664981c3141cddfc59362beb287ebf20d0773660e2dd6faac5968a5930a081` |
| `word/_rels/document.xml.rels` | 1613 | `92ba0b0c1dcef764cd7ba71faa371aa1714446753442b3f3f2804a50a330ff41` |
| `word/styles.xml` | 349845 | `f3cd980374fd3a40a1d79c7d121795d059f9676d6ffdaf746c814d4460a3d22b` |
| `word/stylesWithEffects.xml` | 438131 | `463ae0928cf0d84775dbf8cf18d6c3029f6707c81bf590f6d6dd8757a5e93f15` |
| `word/settings.xml` | 2565 | `86c73de844b13fc27c417004b8fc5a7fdf76224ea4f5bdd0fb3996abedf5c6bb` |
| `word/webSettings.xml` | 438 | `349d36de7434d09f86987ff671d8814964a0588c1e630c06e562cda7e75e9f95` |
| `word/fontTable.xml` | 2811 | `79385fb7f60247507ecaffc292e9ebd52ea0657b8634f629ba6fccc54011d6bb` |
| `word/theme/theme1.xml` | 10939 | `e3a8ab7db9ca7afca56f5f2820a56e8b660016c647773555b060b0a02ac76941` |
| `customXml/item1.xml` | 262 | `a86086ffc5d8e83ebd6c71a55d1d2efaa31b137977f5f3a752366e1023612144` |
| `customXml/_rels/item1.xml.rels` | 295 | `1ca6c9a64edcebe24ee703a54403611b322d96da33371779e742d2d3f7ed7a6c` |
| `customXml/itemProps1.xml` | 354 | `c542307b13ec29a8b546217bb37936ab4822e044b265d2952985ec3d6afed24e` |
| `word/numbering.xml` | 6154 | `b526cd0890f22952741b8cbea471c21ec82152f24f877393b22a91f26c0f2b32` (editable only for abstract numbering definition 91) |
| `word/header1.xml` | 1358 | `01d12af6ae000dbe78c7f7e72cb4fc4bafc1e3a497d228590f93597bda574e55` |
| `word/footer1.xml` | 1371 | `9126574dbb0014fed87e8dc8606bd3a49d07cc902be852cecf7463b347da5797` |
| `word/footer2.xml` | 1371 | `9126574dbb0014fed87e8dc8606bd3a49d07cc902be852cecf7463b347da5797` |
| `docProps/thumbnail.jpeg` | 8324 | `96367138dc44ce09bf2c8f0f8e49348a1478d2c5c0af69bbc2bbc38b63cdcead` |

## Fidelity gates

- The retained reference must remain byte-for-byte unchanged.
- Final document must remain one A4 section and render to exactly four pages.
- Page breaks, table dimensions, borders, fills, numbering definitions, styles, headers, footers, relationships, custom XML, and theme must remain source-derived.
- All English content must be LTR; only the Hebrew research title remains RTL.
- No clipping, overlap, awkward wrapping, orphaned headings, or broken numbering.
- The final output must contain no Hebrew body text other than the required Hebrew title and no unresolved placeholder other than student ID and email.
