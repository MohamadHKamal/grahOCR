"""Batch test runner — process a folder of documents and generate a report.

Usage:
    conda activate graphocr
    python scripts/batch_test.py /path/to/folder
    python scripts/batch_test.py /path/to/folder --output report.html
    python scripts/batch_test.py /path/to/folder --output report.json --format json
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Suppress noisy logs
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


async def run_batch(folder: str, use_surya: bool = False, use_paddle: bool = True) -> list[dict]:
    """Process all documents in folder through the pipeline."""
    from graphocr.pipeline import Pipeline

    pipeline = Pipeline(use_surya=use_surya, use_paddle=use_paddle)
    results = await pipeline.process_batch(folder)
    return [r.summary() for r in results], results


def generate_json_report(summaries: list[dict], output_path: str) -> None:
    """Write JSON report."""
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "total_documents": len(summaries),
        "successful": sum(1 for s in summaries if s["success"]),
        "failed": sum(1 for s in summaries if not s["success"]),
        "aggregate": _compute_aggregate(summaries),
        "documents": summaries,
    }
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"JSON report saved: {output_path}")


def generate_html_report(summaries: list[dict], full_results: list, output_path: str) -> None:
    """Write HTML report with tables and per-document details."""
    agg = _compute_aggregate(summaries)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Build document rows
    doc_rows = ""
    for i, s in enumerate(summaries):
        status = "pass" if s["success"] else "fail"
        status_color = "#2d8a4e" if s["success"] else "#d32f2f"
        failures_str = ", ".join(s.get("failure_types", [])) or "None"
        doc_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td>{Path(s['source']).name}</td>
            <td style="color:{status_color};font-weight:bold">{status.upper()}</td>
            <td>{s['pages']}</td>
            <td>{s['tokens']}</td>
            <td>{s['arabic']}</td>
            <td>{s['english']}</td>
            <td>{s['avg_confidence']:.2%}</td>
            <td>{s['min_confidence']:.2%}</td>
            <td>{s['failures']}</td>
            <td>{failures_str}</td>
            <td>{s['route']}</td>
            <td>{s['uncertainty']:.4f}</td>
            <td>{s['latency_ms']:.0f}ms</td>
            <td style="color:red">{html.escape(s['error'] or '')}</td>
        </tr>"""

    # Build per-document text sections
    doc_details = ""
    for i, r in enumerate(full_results):
        s = summaries[i]
        fname = Path(s['source']).name

        # 1. Full reconstructed text (plain reading order)
        full_text = html.escape(r.full_text_ordered())

        # 2. Per-token annotated text
        text_lines = ""
        for t in sorted(r.tokens, key=lambda x: x.reading_order):
            lang = t.language.value
            conf_pct = f"{t.confidence:.0%}"
            hw = " [HW]" if t.is_handwritten else ""
            zone = f" [{t.zone_label.value}]" if t.zone_label else ""
            conf_color = "#2d8a4e" if t.confidence >= 0.8 else "#e67e22" if t.confidence >= 0.5 else "#d32f2f"
            text_lines += (
                f'<div class="token">'
                f'<span class="lang">[{lang}]</span> '
                f'<span class="conf" style="color:{conf_color}">({conf_pct}){hw}{zone}</span> '
                f'{html.escape(t.text)}'
                f'</div>\n'
            )

        # 3. Token detail table
        token_rows = ""
        for t in sorted(r.tokens, key=lambda x: x.reading_order):
            conf_color = "#2d8a4e" if t.confidence >= 0.8 else "#e67e22" if t.confidence >= 0.5 else "#d32f2f"
            token_rows += (
                f"<tr>"
                f"<td>{t.reading_order}</td>"
                f"<td style='direction:auto;unicode-bidi:plaintext'>{html.escape(t.text)}</td>"
                f"<td>{t.language.value}</td>"
                f"<td style='color:{conf_color};font-weight:600'>{t.confidence:.2%}</td>"
                f"<td>p{t.bbox.page_number}</td>"
                f"<td>({t.bbox.x_min:.0f},{t.bbox.y_min:.0f})-({t.bbox.x_max:.0f},{t.bbox.y_max:.0f})</td>"
                f"<td>{'Yes' if t.is_handwritten else ''}</td>"
                f"<td>{t.zone_label.value if t.zone_label else ''}</td>"
                f"<td><code>{t.token_id[:8]}</code></td>"
                f"</tr>\n"
            )

        # 4. Provenance
        prov_lines = ""
        for t in sorted(r.tokens, key=lambda x: x.reading_order)[:15]:
            prov_lines += f"<div class='prov'>{html.escape(t.to_provenance_str())}</div>\n"

        # 5. Failures
        failure_lines = ""
        for f in r.failures:
            sev_color = "#d32f2f" if f.severity >= 0.7 else "#e67e22" if f.severity >= 0.4 else "#2d8a4e"
            failure_lines += (
                f'<div class="failure">'
                f'<strong style="color:{sev_color}">[{f.failure_type.value}]</strong> '
                f'severity={f.severity:.2f} | remedy={html.escape(f.suggested_remedy)}<br>'
                f'<span style="color:#475569">{html.escape(f.evidence[:200])}</span>'
                f'</div>\n'
            )

        doc_details += f"""
        <div class="doc-detail">
            <h3>{i+1}. {fname}</h3>
            <div class="stats-row">
                <span>Tokens: {s['tokens']}</span>
                <span>Arabic: {s['arabic']}</span>
                <span>English: {s['english']}</span>
                <span>Confidence: {s['avg_confidence']:.2%}</span>
                <span>Route: <strong>{s['route']}</strong></span>
                <span>Uncertainty: {s['uncertainty']:.4f}</span>
                <span>Latency: {s['latency_ms']:.0f}ms</span>
            </div>

            <h4>OCR Result &mdash; Full Text</h4>
            <div class="fulltext-box">{full_text}</div>

            <h4>OCR Result &mdash; Annotated Tokens</h4>
            <div class="text-box">{text_lines}</div>

            <h4>OCR Result &mdash; Token Detail Table</h4>
            <div style="overflow-x:auto">
            <table class="token-table">
                <tr>
                    <th>#</th><th>Text</th><th>Lang</th><th>Confidence</th>
                    <th>Page</th><th>BBox</th><th>HW</th><th>Zone</th><th>ID</th>
                </tr>
                {token_rows}
            </table>
            </div>

            {'<h4>Failures Detected</h4>' + failure_lines if failure_lines else '<p style="color:#2d8a4e;font-weight:600">No failures detected.</p>'}

            <h4>Provenance Trail (first 15)</h4>
            <div class="prov-box">{prov_lines}</div>
        </div>"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GraphOCR Batch Test Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #fafafa; }}
    h1 {{ color: #1a1a2e; }}
    h2 {{ color: #16213e; margin-top: 2rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    h3 {{ color: #0f3460; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.85rem; }}
    th {{ background: #1a1a2e; color: white; padding: 8px 10px; text-align: left; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
    tr:hover {{ background: #f1f5f9; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin: 1rem 0; }}
    .summary-card {{ background: white; border-radius: 8px; padding: 1.2rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .summary-card .label {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase; }}
    .summary-card .value {{ font-size: 1.8rem; font-weight: 700; color: #1a1a2e; }}
    .doc-detail {{ background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .stats-row {{ display: flex; gap: 1.5rem; flex-wrap: wrap; margin: 0.5rem 0; font-size: 0.85rem; color: #475569; }}
    .text-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px; padding: 1rem; max-height: 300px; overflow-y: auto; font-size: 0.85rem; direction: rtl; unicode-bidi: plaintext; }}
    .token {{ padding: 2px 0; direction: ltr; unicode-bidi: plaintext; }}
    .lang {{ color: #6366f1; font-weight: 600; font-size: 0.75rem; }}
    .conf {{ color: #94a3b8; font-size: 0.75rem; }}
    .prov-box {{ background: #1e293b; color: #e2e8f0; border-radius: 4px; padding: 1rem; font-family: monospace; font-size: 0.75rem; max-height: 200px; overflow-y: auto; }}
    .prov {{ padding: 1px 0; }}
    .failure {{ background: #fef2f2; border-left: 3px solid #ef4444; padding: 8px 12px; margin: 6px 0; font-size: 0.82rem; border-radius: 0 4px 4px 0; line-height: 1.5; }}
    .fulltext-box {{ background: #fffbeb; border: 1px solid #fbbf24; border-radius: 4px; padding: 1rem; font-size: 1rem; line-height: 1.8; direction: auto; unicode-bidi: plaintext; white-space: pre-wrap; word-wrap: break-word; max-height: 250px; overflow-y: auto; }}
    .token-table {{ font-size: 0.78rem; }}
    .token-table th {{ background: #334155; font-size: 0.72rem; padding: 6px 8px; }}
    .token-table td {{ padding: 4px 8px; font-size: 0.78rem; }}
    .token-table tr:nth-child(even) {{ background: #f8fafc; }}
</style>
</head>
<body>
<h1>GraphOCR Batch Test Report</h1>
<p style="color:#64748b">Generated: {now} | Documents: {len(summaries)}</p>

<div class="summary-grid">
    <div class="summary-card"><div class="label">Documents</div><div class="value">{len(summaries)}</div></div>
    <div class="summary-card"><div class="label">Successful</div><div class="value" style="color:#2d8a4e">{agg['successful']}</div></div>
    <div class="summary-card"><div class="label">Failed</div><div class="value" style="color:#d32f2f">{agg['failed']}</div></div>
    <div class="summary-card"><div class="label">Total Tokens</div><div class="value">{agg['total_tokens']}</div></div>
    <div class="summary-card"><div class="label">Avg Confidence</div><div class="value">{agg['avg_confidence']:.1%}</div></div>
    <div class="summary-card"><div class="label">Total Failures</div><div class="value">{agg['total_failures']}</div></div>
    <div class="summary-card"><div class="label">VLM Consensus</div><div class="value">{agg['vlm_consensus_count']}</div></div>
    <div class="summary-card"><div class="label">Avg Latency</div><div class="value">{agg['avg_latency_ms']:.0f}ms</div></div>
</div>

<h2>Document Summary Table</h2>
<table>
<tr>
    <th>#</th><th>File</th><th>Status</th><th>Pages</th><th>Tokens</th>
    <th>Arabic</th><th>English</th><th>Avg Conf</th><th>Min Conf</th>
    <th>Failures</th><th>Failure Types</th><th>Route</th><th>Uncertainty</th>
    <th>Latency</th><th>Error</th>
</tr>
{doc_rows}
</table>

<h2>Per-Document Details</h2>
{doc_details}

</body>
</html>"""

    Path(output_path).write_text(html_content, encoding="utf-8")
    print(f"HTML report saved: {output_path}")


def _compute_aggregate(summaries: list[dict]) -> dict:
    """Compute aggregate statistics."""
    successful = [s for s in summaries if s["success"]]
    return {
        "successful": len(successful),
        "failed": len(summaries) - len(successful),
        "total_tokens": sum(s["tokens"] for s in summaries),
        "total_arabic": sum(s["arabic"] for s in summaries),
        "total_english": sum(s["english"] for s in summaries),
        "avg_confidence": sum(s["avg_confidence"] for s in successful) / max(len(successful), 1),
        "min_confidence": min((s["min_confidence"] for s in successful), default=0),
        "total_failures": sum(s["failures"] for s in summaries),
        "cheap_rail_count": sum(1 for s in summaries if s.get("route") == "cheap_rail"),
        "vlm_consensus_count": sum(1 for s in summaries if s.get("route") == "vlm_consensus"),
        "avg_latency_ms": sum(s["latency_ms"] for s in summaries) / max(len(summaries), 1),
    }


def print_console_summary(summaries: list[dict]) -> None:
    """Print summary to console."""
    agg = _compute_aggregate(summaries)

    print(f"\n{'='*70}")
    print(f"  BATCH TEST REPORT")
    print(f"{'='*70}")
    print(f"  Documents:       {len(summaries)}")
    print(f"  Successful:      {agg['successful']}")
    print(f"  Failed:          {agg['failed']}")
    print(f"  Total tokens:    {agg['total_tokens']}")
    print(f"  Arabic tokens:   {agg['total_arabic']}")
    print(f"  English tokens:  {agg['total_english']}")
    print(f"  Avg confidence:  {agg['avg_confidence']:.2%}")
    print(f"  Total failures:  {agg['total_failures']}")
    print(f"  Cheap rail:      {agg['cheap_rail_count']}")
    print(f"  VLM consensus:   {agg['vlm_consensus_count']}")
    print(f"  Avg latency:     {agg['avg_latency_ms']:.0f}ms")
    print(f"{'='*70}")

    for i, s in enumerate(summaries):
        status = "PASS" if s["success"] else "FAIL"
        fname = Path(s["source"]).name
        print(f"  {i+1}. [{status}] {fname:30s} tokens={s['tokens']:3d}  "
              f"conf={s['avg_confidence']:.0%}  failures={s['failures']}  "
              f"route={s['route']:14s}  {s['latency_ms']:.0f}ms")

    print()


def main():
    parser = argparse.ArgumentParser(description="Batch test GraphOCR pipeline")
    parser.add_argument("folder", help="Folder containing document images/PDFs")
    parser.add_argument("--output", "-o", default=None, help="Output report path")
    parser.add_argument("--format", "-f", choices=["html", "json", "both"], default="html",
                        help="Report format (default: html)")
    parser.add_argument("--use-surya", action="store_true", help="Enable Surya Layout Detection")
    parser.add_argument("--use-paddle", action="store_true", help="Enable PaddleOCR Text Detection")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {args.folder} is not a directory")
        sys.exit(1)

    # Default output path
    if args.output is None:
        args.output = str(folder / "graphocr_report")

    print(f"Processing all documents in: {folder}")
    print()

    use_surya = args.use_surya
    use_paddle = args.use_paddle
    # By default, use paddle if no flags are provided
    if not use_surya and not use_paddle:
        use_paddle = True

    summaries, full_results = asyncio.run(run_batch(str(folder), use_surya=use_surya, use_paddle=use_paddle))

    # Console summary always
    print_console_summary(summaries)

    # File reports
    if args.format in ("html", "both"):
        html_path = args.output if args.output.endswith(".html") else args.output + ".html"
        generate_html_report(summaries, full_results, html_path)

    if args.format in ("json", "both"):
        json_path = args.output if args.output.endswith(".json") else args.output + ".json"
        generate_json_report(summaries, json_path)


if __name__ == "__main__":
    main()
