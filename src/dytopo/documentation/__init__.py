"""
DyTopo Documentation Generator
==============================

Auto-generate living documentation from code and execution data.

Features:
- Obsidian canvas generation (visual system map)
- API reference from docstrings
- Architecture diagrams
- Troubleshooting guides

Usage:
    from dytopo.documentation import DocumentationGenerator

    await DocumentationGenerator.generate_overview_canvas(
        Path("docs/overview-obsidian.canvas")
    )

    await DocumentationGenerator.generate_api_reference(
        Path("docs/api-reference.md"),
        AsyncDyTopoOrchestrator
    )
"""

from dytopo.documentation.generator import DocumentationGenerator

__all__ = ["DocumentationGenerator"]
