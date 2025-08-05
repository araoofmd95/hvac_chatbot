#!/usr/bin/env python3
"""
Quick cleanup script for ChromaDB only
For comprehensive cleanup, use cleanup_all.py
"""
import os
import shutil
from pathlib import Path
import sys

def cleanup_chromadb():
    """Remove ChromaDB directories to start fresh"""
    paths_to_clean = [
        Path("chroma_db"),
        Path("src/chroma_db"),
        Path(".chroma"),
        Path("vector_store"),
        Path("src/vector_store"),
    ]
    
    cleaned = False
    for db_path in paths_to_clean:
        if db_path.exists():
            print(f"🗑️  Removing ChromaDB directory: {db_path}")
            shutil.rmtree(db_path)
            cleaned = True
    
    if cleaned:
        print("✅ ChromaDB cleaned successfully!")
    else:
        print("ℹ️  No ChromaDB directories found - already clean")

def main():
    """Run ChromaDB cleanup"""
    print("🧹 ChromaDB Cleanup")
    print("=" * 30)
    
    # Check for --force flag
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        cleanup_chromadb()
    else:
        response = input("Remove ChromaDB data? (y/N): ")
        if response.lower() in ['y', 'yes']:
            cleanup_chromadb()
        else:
            print("❌ Cleanup cancelled")
            return
    
    print("\n💡 Tips:")
    print("   - Use 'python cleanup_db.py --force' to skip confirmation")
    print("   - Use 'python cleanup_all.py' for comprehensive cleanup")
    print("   - The database will be recreated on next document upload")

if __name__ == "__main__":
    main()