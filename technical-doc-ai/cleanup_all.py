#!/usr/bin/env python3
"""
Comprehensive cleanup script for Technical Document AI
Removes all databases, caches, and temporary files
"""
import os
import shutil
from pathlib import Path
import sys

def cleanup_chromadb():
    """Remove ChromaDB directories"""
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
        print("ℹ️  No ChromaDB directories found")

def cleanup_cache():
    """Remove Python cache directories"""
    cache_patterns = ["__pycache__", ".pyc", ".pyo"]
    
    cleaned_count = 0
    for root, dirs, files in os.walk("."):
        # Remove __pycache__ directories
        if "__pycache__" in dirs:
            cache_path = Path(root) / "__pycache__"
            shutil.rmtree(cache_path)
            cleaned_count += 1
        
        # Remove .pyc and .pyo files
        for file in files:
            if file.endswith((".pyc", ".pyo")):
                file_path = Path(root) / file
                file_path.unlink()
                cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"✅ Cleaned {cleaned_count} cache files/directories")
    else:
        print("ℹ️  No cache files found")

def cleanup_logs():
    """Remove log files"""
    log_patterns = ["*.log", "logs/", ".logs/"]
    
    cleaned = False
    
    # Remove log directories
    for log_dir in ["logs", ".logs"]:
        if Path(log_dir).exists():
            print(f"🗑️  Removing log directory: {log_dir}")
            shutil.rmtree(log_dir)
            cleaned = True
    
    # Remove individual log files
    for log_file in Path(".").glob("**/*.log"):
        print(f"🗑️  Removing log file: {log_file}")
        log_file.unlink()
        cleaned = True
    
    if cleaned:
        print("✅ Logs cleaned successfully!")
    else:
        print("ℹ️  No log files found")

def cleanup_temp_files():
    """Remove temporary files"""
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "temp_*",
        "tmp_*",
        ".DS_Store",
        "Thumbs.db",
    ]
    
    cleaned_count = 0
    
    for pattern in temp_patterns:
        for temp_file in Path(".").glob(f"**/{pattern}"):
            print(f"🗑️  Removing temp file: {temp_file}")
            temp_file.unlink()
            cleaned_count += 1
    
    if cleaned_count > 0:
        print(f"✅ Cleaned {cleaned_count} temporary files")
    else:
        print("ℹ️  No temporary files found")

def cleanup_streamlit_cache():
    """Remove Streamlit cache"""
    streamlit_cache_paths = [
        Path(".streamlit/cache"),
        Path("~/.streamlit/cache").expanduser(),
    ]
    
    cleaned = False
    for cache_path in streamlit_cache_paths:
        if cache_path.exists():
            print(f"🗑️  Removing Streamlit cache: {cache_path}")
            shutil.rmtree(cache_path)
            cleaned = True
    
    if cleaned:
        print("✅ Streamlit cache cleaned successfully!")
    else:
        print("ℹ️  No Streamlit cache found")

def cleanup_openai_cache():
    """Remove OpenAI cache if exists"""
    cache_paths = [
        Path(".openai_cache"),
        Path("~/.cache/openai").expanduser(),
    ]
    
    cleaned = False
    for cache_path in cache_paths:
        if cache_path.exists():
            print(f"🗑️  Removing OpenAI cache: {cache_path}")
            shutil.rmtree(cache_path)
            cleaned = True
    
    if cleaned:
        print("✅ OpenAI cache cleaned successfully!")
    else:
        print("ℹ️  No OpenAI cache found")

def main():
    """Run all cleanup operations"""
    print("🧹 Technical Document AI - Comprehensive Cleanup")
    print("=" * 50)
    
    # Confirm before proceeding
    if len(sys.argv) > 1 and sys.argv[1] == "--force":
        confirm = True
    else:
        response = input("⚠️  This will remove all databases and caches. Continue? (y/N): ")
        confirm = response.lower() in ['y', 'yes']
    
    if not confirm:
        print("❌ Cleanup cancelled")
        return
    
    print("\nStarting cleanup...\n")
    
    # Run all cleanup operations
    cleanup_chromadb()
    print()
    
    cleanup_cache()
    print()
    
    cleanup_logs()
    print()
    
    cleanup_temp_files()
    print()
    
    cleanup_streamlit_cache()
    print()
    
    cleanup_openai_cache()
    print()
    
    print("=" * 50)
    print("✨ Cleanup complete!")
    print("\n💡 Tips:")
    print("   - Run 'python run_app.py' to start with a fresh system")
    print("   - Use 'python cleanup_all.py --force' to skip confirmation")
    print("   - The system will recreate necessary databases on next run")

if __name__ == "__main__":
    main()