#!/usr/bin/env python3
"""
Generate all DyTopo documentation.

Generates:
- Overview Obsidian canvas
- API reference
- Troubleshooting guide
- Architecture documentation

Usage:
    python scripts/generate_docs.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dytopo.documentation import DocumentationGenerator


async def main():
    """Generate all documentation."""
    print("Generating DyTopo documentation...")

    docs_dir = Path(__file__).parent.parent / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Generate overview canvas
    print("\n1. Generating overview-obsidian.canvas...")
    await DocumentationGenerator.generate_overview_canvas(
        docs_dir / "overview-obsidian.canvas",
        include_metrics=True,
    )
    print("   [OK] Generated overview-obsidian.canvas")

    # API reference generation currently disabled
    # (would need to import actual module objects, not instances)
    print("\n2. Skipping API reference (manual generation recommended)")

    # Generate troubleshooting guide
    print("\n3. Generating troubleshooting guide...")
    ops_dir = docs_dir / "operations"
    ops_dir.mkdir(exist_ok=True)

    await DocumentationGenerator.generate_troubleshooting_guide(
        ops_dir / "troubleshooting.md"
    )
    print("   [OK] Generated operations/troubleshooting.md")

    # Generate architecture doc
    print("\n4. Generating architecture documentation...")
    arch_dir = docs_dir / "architecture"
    arch_dir.mkdir(exist_ok=True)

    await DocumentationGenerator.generate_architecture_doc(
        arch_dir / "system-overview.md"
    )
    print("   [OK] Generated architecture/system-overview.md")

    print("\n[SUCCESS] All documentation generated successfully!")
    print(f"\nDocumentation location: {docs_dir.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())
