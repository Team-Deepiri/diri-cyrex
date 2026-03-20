#!/bin/bash
# Clean up conflicting packages that require langchain-core 1.x
# These packages are not used in the codebase and conflict with langchain-core 0.2.x

echo "🧹 Cleaning up conflicting packages..."
echo ""

# Uninstall packages that require langchain-core 1.x
echo "Uninstalling packages incompatible with langchain-core 0.2.x:"
echo "  - langchain-classic (requires langchain-core>=1.0.0)"
echo "  - langchain-huggingface (requires langchain-core>=1.1.0)"
echo "  - langgraph-prebuilt (requires langchain-core>=1.0.0)"
echo ""

pip uninstall -y langchain-classic langchain-huggingface langgraph-prebuilt 2>/dev/null || true

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Verifying installation..."
pip check

echo ""
echo "📝 Note: These packages were not in requirements.txt and are not used in the codebase."
echo "   The codebase uses:"
echo "   - langchain_community.vectorstores.Chroma (not langchain-chroma package)"
echo "   - langchain_community.embeddings.HuggingFaceEmbeddings (not langchain-huggingface package)"

