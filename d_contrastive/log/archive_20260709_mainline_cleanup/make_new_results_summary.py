#!/usr/bin/env python
from __future__ import annotations

import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


ROOT = Path("/dcs07/zwang/data/adni_d")
DCON = ROOT / "d_contrastive"
OUT_XLSX = DCON / "AD_contrastive_new_results_summary_20260605.xlsx"
OUT_MD = DCON / "AD_contrastive_new_results_summary_20260605.md"
OUT_KEY_CSV = DCON / "AD_contrastive_key_results_20260605.csv"


def is_number(x):
    if x is None or pd.isna(x):
        return False
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def col_name(n):
    name = ""
    n += 1
    while n:
        n, rem = divmod(n - 1, 26)
        name = chr(65 + rem) + name
    return name


def sheet_xml(df: pd.DataFrame) -> str:
    rows = []
    header = list(df.columns)
    all_rows = [header] + df.where(pd.notna(df), "").values.tolist()
    for r_idx, row in enumerate(all_rows, start=1):
        cells = []
        for c_idx, value in enumerate(row):
            ref = f"{col_name(c_idx)}{r_idx}"
            if is_number(value):
                cells.append(f'<c r="{ref}"><v>{float(value):.10g}</v></c>')
            elif value == "":
                cells.append(f'<c r="{ref}"/>')
            else:
                text = escape(str(value))
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>')
        rows.append(f'<row r="{r_idx}">{"".join(cells)}</row>')
    dim = f"A1:{col_name(max(len(header) - 1, 0))}{len(all_rows)}"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<dimension ref="{dim}"/>'
        '<sheetViews><sheetView workbookViewId="0"/></sheetViews>'
        '<sheetFormatPr defaultRowHeight="15"/>'
        f'<sheetData>{"".join(rows)}</sheetData>'
        '</worksheet>'
    )


def write_xlsx(sheets: dict[str, pd.DataFrame], path: Path):
    sheet_names = list(sheets)
    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>',
    ]
    for i in range(1, len(sheet_names) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append("</Types>")

    workbook_sheets = []
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]
    for i, name in enumerate(sheet_names, start=1):
        workbook_sheets.append(f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        workbook_rels.append(
            f'<Relationship Id="rId{i}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
        )
    workbook_rels.append(
        f'<Relationship Id="rId{len(sheet_names)+1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/>'
    )
    workbook_rels.append("</Relationships>")

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        '</workbook>'
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        '</Relationships>'
    )
    styles = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>'
        '<fills count="1"><fill><patternFill patternType="none"/></fill></fills>'
        '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
        '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
        '<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>'
        '</styleSheet>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "".join(content_types))
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", "".join(workbook_rels))
        zf.writestr("xl/styles.xml", styles)
        for i, name in enumerate(sheet_names, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", sheet_xml(sheets[name]))


def main():
    baseline = pd.read_csv(DCON / "rank_sweep_with_ml_baselines.csv")
    best = pd.read_csv(DCON / "rank_sweep_best_comparison.csv")
    exp0 = pd.read_csv(DCON / "exp0_dhat_vs_contrastive_conversion.csv")
    exp1 = pd.read_csv(DCON / "exp1_method2_dx_pretrain.csv")
    exp4 = pd.read_csv(DCON / "exp4_conversion_task_suite.csv")
    counts = pd.read_csv(DCON / "exp4_conversion_cohort_counts.csv")
    counts["n_excluded_short_followup"] = counts["n_before_censor"] - counts["n_after_censor"]
    counts.to_csv(DCON / "exp4_conversion_cohort_counts.csv", index=False)

    readme = pd.DataFrame(
        [
            ["Dataset", "Frozen 128^3 Swin embeddings, 3019 images, train 2408 images / test 611 images."],
            ["Rule", "Swin encoder was not trained. All experiments use frozen 768-d embeddings."],
            ["Best current dx + d_mod3 baseline", "Pure ML direct LR/Ridge on raw 768: CN/AD AUC 0.860, MCI/AD AUC 0.762, d_mod3 R2 0.237."],
            ["Contrastive value", "Euclidean/hybrid improve conversion tasks, especially sparse CN->MCI 2y, but not d_mod3 regression."],
            ["Dense supervision finding", "Ridge d_hat explains much of conversion gain; this supports dense progression supervision more than uniquely contrastive geometry."],
            ["dx-pretrain finding", "dx-pretrain Setup3 is close to euclidean contrastive on MCI->AD, so contrastive objective is not uniquely better."],
            ["Power caveat", "CN->MCI 2y has only 5 test converters; 3y/4y have 9 converters. Treat conversion AUC as directional."],
            ["CI note", "EXP4 quick CI used N_BOOT=200 for speed. For manuscript, rerun EXP4 with N_BOOT=5000."],
        ],
        columns=["item", "summary"],
    )

    key_rows = [
        baseline.assign(source_table="main_baselines"),
        exp1.rename(columns={
            "cn_ad_auc_mean": "cn_ad_auc",
            "cn_mci_cls_auc_mean": "cn_mci_cls_auc",
            "mci_ad_cls_auc_mean": "mci_ad_cls_auc",
            "3class_macro_auc_mean": "3class_macro_auc",
            "mci_conv_auc_mean": "mci_conv_auc",
            "cn_mci_conv_auc_mean": "cn_mci_conv_auc",
            "r2_d_mod3_mean": "r2_d_mod3",
            "pearson_d_mod3_mean": "pearson_d_mod3",
            "spearman_d_mod3_mean": "spearman_d_mod3",
            "z_std_test_mean": "z_std_test",
            "z_active_dims_005_mean": "z_active_dims_005",
        }).assign(method=lambda d: d["setup"], source_table="dx_pretrain").drop(columns=["setup"], errors="ignore"),
    ]
    key = pd.concat(key_rows, ignore_index=True, sort=False)
    preferred_cols = [
        "source_table", "method", "input_version", "n_seeds",
        "cn_ad_auc", "cn_mci_cls_auc", "mci_ad_cls_auc", "3class_macro_auc",
        "mci_conv_auc", "cn_mci_conv_auc", "r2_d_mod3", "pearson_d_mod3",
        "spearman_d_mod3", "z_std_test", "z_active_dims_005",
    ]
    key = key[[c for c in preferred_cols if c in key.columns]]
    key.to_csv(OUT_KEY_CSV, index=False)

    sheets = {
        "README": readme,
        "key_summary": key,
        "main_baselines": baseline,
        "best_rank_compare": best,
        "exp0_dhat_conversion": exp0,
        "exp1_dx_pretrain": exp1,
        "exp4_conversion_windows": exp4,
        "exp4_cohort_counts": counts,
    }
    write_xlsx(sheets, OUT_XLSX)

    md = [
        "# AD contrastive new results summary",
        "",
        "## Files",
        f"- Excel workbook: `{OUT_XLSX}`",
        f"- Key CSV: `{OUT_KEY_CSV}`",
        "",
        "## Key read",
        "- Direct LR/Ridge on raw 768 is strongest for current diagnosis and d_mod3 regression.",
        "- Euclidean/hybrid contrastive mainly helps conversion, but conversion cohorts are small.",
        "- Ridge d_hat explains much of the conversion signal, so the safest claim is dense progression supervision rather than contrastive-specific geometry.",
        "- dx-pretrain Setup3 is close to Euclidean contrastive, so contrastive does not clearly beat supervised-dx pretraining.",
        "",
        "## Main baseline table",
        baseline.to_markdown(index=False),
        "",
        "## EXP1 dx-pretrain",
        exp1.to_markdown(index=False),
        "",
        "## EXP0 d_hat conversion",
        exp0.to_markdown(index=False),
        "",
    ]
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_XLSX}")
    print(f"Wrote {OUT_KEY_CSV}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
