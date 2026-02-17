#!/bin/bash
# Stop AnyLoom Docker Stack

echo "============================================"
echo "  Stopping AnyLoom Docker Stack"
echo "============================================"
echo ""

docker compose down

echo ""
echo "Services stopped."
echo ""
echo "To remove volumes (WARNING: deletes all data):"
echo "  docker compose down -v"
echo ""
