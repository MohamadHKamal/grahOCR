"""CLI entry point for the GraphOCR pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from graphocr.core.config import get_settings
from graphocr.core.logging import setup_logging

app = typer.Typer(name="graphocr", help="Hybrid Graph-OCR Pipeline CLI")


@app.command()
def process(
    file_path: str = typer.Argument(..., help="Path to document (PDF, TIFF, JPEG, PNG)"),
    output: str = typer.Option("stdout", help="Output path or 'stdout'"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Process a single insurance claim document."""
    setup_logging("DEBUG" if verbose else "INFO")
    asyncio.run(_process_file(file_path, output))


async def _process_file(file_path: str, output: str) -> None:
    from graphocr.layer1_foundation.ingestion import load_document
    from graphocr.layer1_foundation.language_detector import assign_languages
    from graphocr.layer1_foundation.ocr_paddleocr import PaddleOCREngine
    from graphocr.layer1_foundation.reading_order import assign_reading_order
    from graphocr.layer1_foundation.spatial_assembler import assemble_tokens
    from graphocr.layer3_inference.cheap_rail import process_cheap_rail
    from graphocr.layer3_inference.traffic_controller import route_document
    from graphocr.models.document import RawDocument

    path = Path(file_path)
    if not path.exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    doc = RawDocument(
        source_path=str(path),
        file_format=path.suffix.lstrip(".").lower(),
        file_size_bytes=path.stat().st_size,
    )

    typer.echo(f"Processing: {path.name}")

    pages = load_document(doc, f"/tmp/graphocr/{doc.document_id}")
    typer.echo(f"  Pages: {len(pages)}")

    ocr = PaddleOCREngine()
    token_streams = [ocr.extract(page) for page in pages]
    tokens = assemble_tokens(token_streams)
    tokens = assign_reading_order(tokens)
    tokens = assign_languages(tokens)
    typer.echo(f"  Tokens: {len(tokens)}")

    routing = route_document(tokens)
    typer.echo(f"  Route: {routing.path.value} (uncertainty: {routing.uncertainty_score:.3f})")

    extraction = await process_cheap_rail(doc.document_id, tokens)
    typer.echo(f"  Confidence: {extraction.overall_confidence:.3f}")

    result = {
        "document_id": doc.document_id,
        "processing_path": routing.path.value,
        "fields": {k: {"value": v.value, "confidence": v.confidence} for k, v in extraction.fields.items()},
        "overall_confidence": extraction.overall_confidence,
    }

    if output == "stdout":
        typer.echo(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        typer.echo(f"  Output: {output}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind"),
    port: int = typer.Option(8080, help="Port to listen on"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
):
    """Start the FastAPI server."""
    import uvicorn

    setup_logging("INFO")
    uvicorn.run("graphocr.app:app", host=host, port=port, reload=reload)


@app.command()
def seed_graph():
    """Seed Neo4j with domain rules from config."""
    setup_logging("INFO")
    asyncio.run(_seed())


async def _seed() -> None:
    from graphocr.layer2_verification.knowledge_graph.client import Neo4jClient
    from graphocr.layer2_verification.knowledge_graph.schema_loader import load_schema

    client = Neo4jClient()
    await client.connect()
    try:
        await load_schema(client)
        typer.echo("Neo4j schema seeded successfully.")
    finally:
        await client.close()


@app.command()
def supervisor_status():
    """Check DSPy supervisor status."""
    from graphocr.dspy_layer.supervisor import DSPySupervisor

    supervisor = DSPySupervisor()
    status = supervisor.get_status()
    typer.echo(json.dumps(status, indent=2, default=str))


@app.command()
def test(
    unit: bool = typer.Option(False, "--unit", help="Run unit tests only"),
    integration: bool = typer.Option(False, "--integration", help="Run integration tests only"),
    layer1: bool = typer.Option(False, "--layer1", help="Run Layer 1 (OCR Foundation) tests"),
    layer2: bool = typer.Option(False, "--layer2", help="Run Layer 2 (Verification) tests"),
    layer3: bool = typer.Option(False, "--layer3", help="Run Layer 3 (Inference) tests"),
    rag: bool = typer.Option(False, "--rag", help="Run RAG retriever tests"),
    dspy: bool = typer.Option(False, "--dspy", help="Run DSPy tests"),
    smoke: bool = typer.Option(False, "--smoke", help="Run fast smoke tests only"),
    cov: bool = typer.Option(False, "--cov", help="Run with coverage report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Run the test suite with optional filters.

    Examples:
        graphocr test                    # all tests
        graphocr test --unit             # unit tests only
        graphocr test --layer1 --layer2  # Layer 1 + Layer 2
        graphocr test --smoke --cov      # smoke tests with coverage
    """
    import subprocess
    import sys

    args = ["python", "-m", "pytest", "tests/"]

    markers: list[str] = []
    if unit:
        markers.append("unit")
    if integration:
        markers.append("integration")
    if layer1:
        markers.append("layer1")
    if layer2:
        markers.append("layer2")
    if layer3:
        markers.append("layer3")
    if rag:
        markers.append("rag")
    if dspy:
        markers.append("dspy")
    if smoke:
        markers.append("smoke")

    if markers:
        args.extend(["-m", " or ".join(markers)])

    if cov:
        args.extend(["--cov=graphocr", "--cov-report=term-missing"])

    if verbose:
        args.append("-v")

    result = subprocess.run(args)
    sys.exit(result.returncode)


@app.command(name="test-pipeline")
def test_pipeline(
    file_path: str = typer.Argument(..., help="Path to document (PDF, TIFF, JPEG, PNG)"),
    surya: bool = typer.Option(False, "--surya/--no-surya", help="Enable Surya layout engine"),
    paddle: bool = typer.Option(True, "--paddle/--no-paddle", help="Enable PaddleOCR engine"),
    all_engines: bool = typer.Option(False, "--all-engines", help="Run all engine combos and compare"),
    output: str = typer.Option(None, "--output", "-o", help="Report output path (auto-generates if omitted)"),
    format: str = typer.Option("both", "--format", "-f", help="Report format: html, json, or both"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Test the pipeline on a document with different OCR engine configurations.

    Generates an HTML + JSON report with full details, diagnostics, and suggestions.

    Examples:
        graphocr test-pipeline claim.pdf                     # PaddleOCR only (default)
        graphocr test-pipeline claim.pdf --surya              # PaddleOCR + Surya
        graphocr test-pipeline claim.pdf --no-paddle --surya  # Surya only
        graphocr test-pipeline claim.pdf --all-engines        # Compare all 3 combos
        graphocr test-pipeline claim.pdf -o report -f both    # Custom output path
    """
    setup_logging("DEBUG" if verbose else "INFO")

    if all_engines:
        configs = [
            ("PaddleOCR only", True, False),
            ("PaddleOCR + Surya", True, True),
            ("Surya only", False, True),
        ]
    else:
        label = []
        if paddle:
            label.append("PaddleOCR")
        if surya:
            label.append("Surya")
        if not label:
            typer.echo("Error: at least one engine must be enabled (--paddle or --surya)", err=True)
            raise typer.Exit(1)
        configs = [(" + ".join(label), paddle, surya)]

    if output is None:
        output = str(Path(file_path).parent / f"pipeline_test_{Path(file_path).stem}")

    asyncio.run(_run_pipeline_test(file_path, configs, verbose, output, format))


async def _run_pipeline_test(
    file_path: str,
    configs: list[tuple[str, bool, bool]],
    verbose: bool,
    output_path: str,
    report_format: str,
) -> None:
    from datetime import datetime

    from graphocr.pipeline import Pipeline

    path = Path(file_path)
    if not path.exists():
        typer.echo(f"Error: File not found: {file_path}", err=True)
        raise typer.Exit(1)

    test_start = datetime.now()
    run_results: list[tuple[str, object]] = []

    for label, use_paddle, use_surya in configs:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  {label}")
        typer.echo(f"{'='*60}\n")

        pipeline = Pipeline(use_paddle=use_paddle, use_surya=use_surya, use_rag=False)
        result = await pipeline.process(str(path))

        if result.error:
            typer.echo(f"  ERROR: {result.error}")
            run_results.append((label, result))
            continue

        typer.echo(f"  Pages:          {len(result.pages)}")
        typer.echo(f"  Tokens:         {result.total_tokens}")
        typer.echo(f"  Arabic tokens:  {result.arabic_tokens}")
        typer.echo(f"  English tokens: {result.english_tokens}")
        typer.echo(f"  Avg confidence: {result.avg_confidence:.4f}")
        typer.echo(f"  Min confidence: {result.min_confidence:.4f}")
        typer.echo(f"  Max confidence: {result.max_confidence:.4f}")
        typer.echo(f"  Failures:       {len(result.failures)}")
        if result.routing:
            typer.echo(f"  Route:          {result.routing.path.value}")
            typer.echo(f"  Uncertainty:    {result.routing.uncertainty_score:.4f}")
        typer.echo(f"  Latency:        {result.latency_ms:.0f} ms")

        if verbose and result.failures:
            typer.echo(f"\n  Failures:")
            for f in result.failures:
                typer.echo(f"    - [{f.failure_type.value}] severity={f.severity:.2f}: {f.evidence[:80]}")

        if verbose:
            typer.echo(f"\n  First 10 tokens:")
            for t in sorted(result.tokens, key=lambda x: x.reading_order)[:10]:
                typer.echo(f"    [{t.language.value}] ({t.confidence:.0%}) {t.text}")

        run_results.append((label, result))

    # Comparison table (console)
    if len(run_results) > 1:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  COMPARISON")
        typer.echo(f"{'='*60}\n")
        typer.echo(f"  {'Engine':<25} {'Tokens':>7} {'Avg Conf':>9} {'Failures':>9} {'Latency':>10}")
        typer.echo(f"  {'-'*25} {'-'*7} {'-'*9} {'-'*9} {'-'*10}")
        for label, r in run_results:
            if r.error:
                typer.echo(f"  {label:<25} {'ERROR':>7}")
            else:
                typer.echo(
                    f"  {label:<25} {r.total_tokens:>7} {r.avg_confidence:>9.4f}"
                    f" {len(r.failures):>9} {r.latency_ms:>8.0f}ms"
                )

    # Generate suggestions
    suggestions = _generate_suggestions(run_results)
    if suggestions:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  SUGGESTIONS")
        typer.echo(f"{'='*60}\n")
        for s in suggestions:
            typer.echo(f"  {s}")

    # Generate reports
    test_end = datetime.now()
    report_data = _build_report_data(file_path, configs, run_results, suggestions, test_start, test_end)

    if report_format in ("json", "both"):
        json_path = output_path if output_path.endswith(".json") else output_path + ".json"
        Path(json_path).write_text(json.dumps(report_data, indent=2, ensure_ascii=False, default=str))
        typer.echo(f"\n  JSON report: {json_path}")

    if report_format in ("html", "both"):
        html_path = output_path if output_path.endswith(".html") else output_path + ".html"
        _write_html_report(report_data, run_results, html_path)
        typer.echo(f"  HTML report: {html_path}")


def _generate_suggestions(run_results: list[tuple[str, object]]) -> list[str]:
    """Generate actionable suggestions based on test results."""
    suggestions = []

    for label, r in run_results:
        if r.error:
            suggestions.append(f"[{label}] Pipeline failed: {r.error}")
            continue

        # Low confidence
        if r.avg_confidence < 0.5:
            suggestions.append(
                f"[{label}] Low avg confidence ({r.avg_confidence:.2%}) — document is likely "
                f"handwritten or degraded. Consider enabling Surya for zone detection and "
                f"confidence boosting via dual-engine merge."
            )
        elif r.avg_confidence < 0.7:
            suggestions.append(
                f"[{label}] Moderate confidence ({r.avg_confidence:.2%}) — review image quality. "
                f"Stamp suppression, lighting correction, or higher DPI scan may help."
            )

        # High failure count
        if len(r.failures) >= 3:
            suggestions.append(
                f"[{label}] Multiple failures detected ({len(r.failures)}) — document layout "
                f"may have complex multi-column structure or overlapping stamps."
            )

        # Specific failure types
        for f in r.failures:
            if "cross_column" in f.evidence.lower():
                suggestions.append(
                    f"[{label}] Cross-column merge detected — XY-Cut may need calibration "
                    f"for this document layout. Try with Surya for better zone segmentation."
                )
            if "stamp" in f.evidence.lower():
                suggestions.append(
                    f"[{label}] Stamp/seal overlap — stamp suppression caught some but check "
                    f"if underlying text was preserved. VLM re-scan would target this region."
                )

        # VLM routing
        if r.routing and r.routing.path.value == "vlm_consensus":
            suggestions.append(
                f"[{label}] Routed to VLM Consensus (uncertainty={r.routing.uncertainty_score:.3f}) — "
                f"this document would use the full adversarial pipeline in production."
            )

        # Language mixing
        if r.arabic_tokens > 0 and r.english_tokens > 0:
            total = r.arabic_tokens + r.english_tokens
            ar_pct = r.arabic_tokens / total * 100
            suggestions.append(
                f"[{label}] Bilingual document ({ar_pct:.0f}% Arabic, {100-ar_pct:.0f}% English) — "
                f"RTL reading order is active. Verify Arabic text reads right-to-left."
            )

        # No tokens
        if r.total_tokens == 0:
            suggestions.append(
                f"[{label}] Zero tokens extracted — document may be blank, corrupted, "
                f"or in an unsupported format. Check input file."
            )

        # Very high latency
        if r.latency_ms > 60000:
            suggestions.append(
                f"[{label}] High latency ({r.latency_ms/1000:.0f}s) — consider GPU acceleration "
                f"or reducing max image size for faster processing."
            )

    # Cross-engine comparison suggestions
    if len(run_results) > 1:
        valid = [(l, r) for l, r in run_results if not r.error and r.total_tokens > 0]
        if len(valid) >= 2:
            best_conf = max(valid, key=lambda x: x[1].avg_confidence)
            best_tokens = max(valid, key=lambda x: x[1].total_tokens)
            fastest = min(valid, key=lambda x: x[1].latency_ms)

            if best_conf[0] != fastest[0]:
                suggestions.append(
                    f"Best accuracy: {best_conf[0]} ({best_conf[1].avg_confidence:.2%}). "
                    f"Fastest: {fastest[0]} ({fastest[1].latency_ms:.0f}ms). "
                    f"Choose based on your accuracy vs speed tradeoff."
                )
            if best_tokens[1].total_tokens > min(v[1].total_tokens for v in valid) * 1.2:
                suggestions.append(
                    f"{best_tokens[0]} extracted the most tokens ({best_tokens[1].total_tokens}) — "
                    f"dual-engine mode may recover text missed by a single engine."
                )

    return suggestions


def _build_report_data(
    file_path: str,
    configs: list[tuple[str, bool, bool]],
    run_results: list[tuple[str, object]],
    suggestions: list[str],
    test_start,
    test_end,
) -> dict:
    """Build structured report data for JSON/HTML output."""
    runs = []
    for (label, use_paddle, use_surya), (_, result) in zip(configs, run_results):
        run_data = {
            "engine": label,
            "use_paddle": use_paddle,
            "use_surya": use_surya,
            "error": result.error,
        }
        if not result.error:
            run_data.update({
                "document_id": result.document_id,
                "pages": len(result.pages),
                "total_tokens": result.total_tokens,
                "arabic_tokens": result.arabic_tokens,
                "english_tokens": result.english_tokens,
                "avg_confidence": round(result.avg_confidence, 4),
                "min_confidence": round(result.min_confidence, 4),
                "max_confidence": round(result.max_confidence, 4),
                "failures": len(result.failures),
                "failure_details": [
                    {
                        "type": f.failure_type.value,
                        "severity": round(f.severity, 3),
                        "remedy": f.suggested_remedy,
                        "evidence": f.evidence[:200],
                    }
                    for f in result.failures
                ],
                "route": result.routing.path.value if result.routing else None,
                "uncertainty": round(result.routing.uncertainty_score, 4) if result.routing else None,
                "latency_ms": round(result.latency_ms, 1),
                "tokens": [
                    {
                        "id": t.token_id[:8],
                        "text": t.text,
                        "language": t.language.value,
                        "confidence": round(t.confidence, 4),
                        "page": t.bbox.page_number,
                        "bbox": f"({t.bbox.x_min:.0f},{t.bbox.y_min:.0f})-({t.bbox.x_max:.0f},{t.bbox.y_max:.0f})",
                        "is_handwritten": t.is_handwritten,
                        "zone": t.zone_label.value if t.zone_label else None,
                        "engine": t.ocr_engine,
                    }
                    for t in sorted(result.tokens, key=lambda x: x.reading_order)
                ],
                "provenance": [
                    t.to_provenance_str()
                    for t in sorted(result.tokens, key=lambda x: x.reading_order)[:20]
                ],
            })
        runs.append(run_data)

    return {
        "report_type": "pipeline_test",
        "generated_at": test_end.isoformat(),
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "test_duration_seconds": round((test_end - test_start).total_seconds(), 1),
        "source_file": str(Path(file_path).resolve()),
        "file_name": Path(file_path).name,
        "file_size_bytes": Path(file_path).stat().st_size if Path(file_path).exists() else 0,
        "engine_configurations": len(configs),
        "runs": runs,
        "suggestions": suggestions,
    }


def _write_html_report(report_data: dict, run_results: list[tuple[str, object]], html_path: str) -> None:
    """Generate a detailed HTML report."""
    import html as html_mod

    now = report_data["generated_at"]
    fname = report_data["file_name"]
    duration = report_data["test_duration_seconds"]

    # Build run cards
    run_cards = ""
    for i, run in enumerate(report_data["runs"]):
        label = run["engine"]
        if run.get("error"):
            run_cards += f"""
            <div class="run-card" style="border-left:4px solid #d32f2f">
                <h3>{label}</h3>
                <p style="color:#d32f2f">ERROR: {html_mod.escape(run['error'])}</p>
            </div>"""
            continue

        route_color = "#2d8a4e" if run["route"] == "cheap_rail" else "#e67e22"
        conf_color = "#2d8a4e" if run["avg_confidence"] >= 0.8 else "#e67e22" if run["avg_confidence"] >= 0.5 else "#d32f2f"

        # Token table
        token_rows = ""
        for t in run.get("tokens", []):
            tc = "#2d8a4e" if t["confidence"] >= 0.8 else "#e67e22" if t["confidence"] >= 0.5 else "#d32f2f"
            token_rows += (
                f"<tr>"
                f"<td style='direction:auto;unicode-bidi:plaintext'>{html_mod.escape(t['text'])}</td>"
                f"<td>{t['language']}</td>"
                f"<td style='color:{tc};font-weight:600'>{t['confidence']:.2%}</td>"
                f"<td>p{t['page']}</td>"
                f"<td>{t['bbox']}</td>"
                f"<td>{'Yes' if t['is_handwritten'] else ''}</td>"
                f"<td>{t['zone'] or ''}</td>"
                f"<td>{t['engine']}</td>"
                f"<td><code>{t['id']}</code></td>"
                f"</tr>"
            )

        # Failure details
        failure_html = ""
        if run["failure_details"]:
            for f in run["failure_details"]:
                sc = "#d32f2f" if f["severity"] >= 0.7 else "#e67e22" if f["severity"] >= 0.4 else "#2d8a4e"
                failure_html += (
                    f'<div class="failure">'
                    f'<strong style="color:{sc}">[{f["type"]}]</strong> '
                    f'severity={f["severity"]:.2f} | remedy={html_mod.escape(f["remedy"])}<br>'
                    f'<span style="color:#475569">{html_mod.escape(f["evidence"])}</span>'
                    f'</div>'
                )
        else:
            failure_html = '<p style="color:#2d8a4e;font-weight:600">No failures detected.</p>'

        # Provenance
        prov_html = ""
        for p in run.get("provenance", []):
            prov_html += f"<div class='prov'>{html_mod.escape(p)}</div>"

        run_cards += f"""
        <div class="run-card">
            <h3>{label}</h3>
            <div class="stats-grid">
                <div class="stat"><div class="stat-label">Tokens</div><div class="stat-value">{run['total_tokens']}</div></div>
                <div class="stat"><div class="stat-label">Arabic</div><div class="stat-value">{run['arabic_tokens']}</div></div>
                <div class="stat"><div class="stat-label">English</div><div class="stat-value">{run['english_tokens']}</div></div>
                <div class="stat"><div class="stat-label">Avg Confidence</div><div class="stat-value" style="color:{conf_color}">{run['avg_confidence']:.2%}</div></div>
                <div class="stat"><div class="stat-label">Min Confidence</div><div class="stat-value">{run['min_confidence']:.2%}</div></div>
                <div class="stat"><div class="stat-label">Max Confidence</div><div class="stat-value">{run['max_confidence']:.2%}</div></div>
                <div class="stat"><div class="stat-label">Failures</div><div class="stat-value">{run['failures']}</div></div>
                <div class="stat"><div class="stat-label">Route</div><div class="stat-value" style="color:{route_color}">{run['route']}</div></div>
                <div class="stat"><div class="stat-label">Uncertainty</div><div class="stat-value">{run['uncertainty']:.4f}</div></div>
                <div class="stat"><div class="stat-label">Latency</div><div class="stat-value">{run['latency_ms']:.0f}ms</div></div>
            </div>

            <h4>Failures</h4>
            {failure_html}

            <h4>Token Detail Table</h4>
            <div style="overflow-x:auto">
            <table class="token-table">
                <tr><th>Text</th><th>Lang</th><th>Confidence</th><th>Page</th><th>BBox</th><th>HW</th><th>Zone</th><th>Engine</th><th>ID</th></tr>
                {token_rows}
            </table>
            </div>

            <h4>Provenance Trail (first 20)</h4>
            <div class="prov-box">{prov_html}</div>
        </div>"""

    # Comparison table (if multiple runs)
    comparison_html = ""
    if len(report_data["runs"]) > 1:
        comp_rows = ""
        for run in report_data["runs"]:
            if run.get("error"):
                comp_rows += f"<tr><td>{run['engine']}</td><td colspan='6' style='color:#d32f2f'>ERROR</td></tr>"
            else:
                comp_rows += (
                    f"<tr><td>{run['engine']}</td><td>{run['total_tokens']}</td>"
                    f"<td>{run['avg_confidence']:.2%}</td><td>{run['failures']}</td>"
                    f"<td>{run['route']}</td><td>{run['uncertainty']:.4f}</td>"
                    f"<td>{run['latency_ms']:.0f}ms</td></tr>"
                )
        comparison_html = f"""
        <h2>Engine Comparison</h2>
        <table><tr><th>Engine</th><th>Tokens</th><th>Avg Conf</th><th>Failures</th><th>Route</th><th>Uncertainty</th><th>Latency</th></tr>
        {comp_rows}</table>"""

    # Suggestions
    suggestions_html = ""
    if report_data["suggestions"]:
        items = "".join(f"<li>{html_mod.escape(s)}</li>" for s in report_data["suggestions"])
        suggestions_html = f"<h2>Suggestions</h2><ul class='suggestions'>{items}</ul>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GraphOCR Pipeline Test Report — {html_mod.escape(fname)}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 2rem; background: #fafafa; color: #1a1a2e; }}
    h1 {{ color: #1a1a2e; margin-bottom: 0.3rem; }}
    h2 {{ color: #16213e; margin-top: 2rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    h3 {{ color: #0f3460; margin-top: 0; }}
    h4 {{ color: #334155; margin-top: 1.2rem; }}
    .meta {{ color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .meta span {{ margin-right: 2rem; }}
    .run-card {{ background: white; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border-left: 4px solid #3b82f6; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.8rem; margin: 1rem 0; }}
    .stat {{ background: #f8fafc; border-radius: 6px; padding: 0.7rem; text-align: center; }}
    .stat-label {{ font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }}
    .stat-value {{ font-size: 1.3rem; font-weight: 700; color: #1a1a2e; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.82rem; }}
    th {{ background: #1a1a2e; color: white; padding: 8px 10px; text-align: left; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #e2e8f0; }}
    tr:hover {{ background: #f1f5f9; }}
    .token-table {{ font-size: 0.78rem; }}
    .token-table th {{ background: #334155; font-size: 0.72rem; padding: 6px 8px; }}
    .token-table td {{ padding: 4px 8px; }}
    .token-table tr:nth-child(even) {{ background: #f8fafc; }}
    .prov-box {{ background: #1e293b; color: #e2e8f0; border-radius: 4px; padding: 1rem; font-family: monospace; font-size: 0.75rem; max-height: 250px; overflow-y: auto; }}
    .prov {{ padding: 1px 0; }}
    .failure {{ background: #fef2f2; border-left: 3px solid #ef4444; padding: 8px 12px; margin: 6px 0; font-size: 0.82rem; border-radius: 0 4px 4px 0; line-height: 1.5; }}
    .suggestions {{ background: #fffbeb; border: 1px solid #f59e0b; border-radius: 8px; padding: 1rem 1rem 1rem 2rem; font-size: 0.88rem; line-height: 1.8; }}
    .suggestions li {{ margin-bottom: 0.5rem; }}
</style>
</head>
<body>
<h1>GraphOCR Pipeline Test Report</h1>
<div class="meta">
    <span>File: <strong>{html_mod.escape(fname)}</strong></span>
    <span>Date: <strong>{now}</strong></span>
    <span>Duration: <strong>{duration}s</strong></span>
    <span>Engine configs: <strong>{report_data['engine_configurations']}</strong></span>
    <span>File size: <strong>{report_data['file_size_bytes']:,} bytes</strong></span>
</div>

{comparison_html}
{suggestions_html}

<h2>Detailed Results</h2>
{run_cards}

<footer style="margin-top:3rem;padding-top:1rem;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.8rem">
    Generated by GraphOCR Pipeline Test | {now}
</footer>
</body>
</html>"""

    Path(html_path).write_text(html_content, encoding="utf-8")


if __name__ == "__main__":
    app()
