#!/usr/bin/env python3
"""Clean up debug logs and test artifacts while preserving important files"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta


def setup_logging():
    """Setup logging for cleanup script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def move_test_files_to_tests_dir(project_root: Path, logger):
    """Move test files from root to tests directory"""
    test_files = [
        'test_measure_debug.py',
        'test_measure_fix.py', 
        'test_osh_validation.py',
        'test_all_validation_programs.py',
        'test_dynamic_validation.py',
        'test_dynamic_validation_quick.py',
        'test_websocket_node.js'
    ]
    
    tests_dir = project_root / 'tests' / 'debug_moved'
    tests_dir.mkdir(parents=True, exist_ok=True)
    
    moved_count = 0
    for test_file in test_files:
        src = project_root / test_file
        if src.exists():
            dst = tests_dir / test_file
            logger.info(f"Moving {test_file} to {tests_dir}")
            shutil.move(str(src), str(dst))
            moved_count += 1
            
    logger.info(f"Moved {moved_count} test files to {tests_dir}")
    return moved_count


def organize_debug_html_files(project_root: Path, logger):
    """Organize debug HTML files in frontend/public"""
    public_dir = project_root / 'frontend' / 'public'
    debug_html_dir = public_dir / 'debug_html'
    debug_html_dir.mkdir(exist_ok=True)
    
    debug_patterns = [
        'test-websocket-debug.html',
        'test-websocket-simple.html', 
        'test-universe-metrics.html',
        'test-universe-metrics-debug.html',
        'test-universe-metrics-flow.html',
        'test-*.html',  # Other test HTML files
        'debug.html',
        'check-console.html',
        'console-*.html'
    ]
    
    moved_count = 0
    for pattern in debug_patterns:
        if '*' in pattern:
            # Handle wildcard patterns
            prefix = pattern.replace('*.html', '')
            for html_file in public_dir.glob(f"{prefix}*.html"):
                if html_file.is_file():
                    dst = debug_html_dir / html_file.name
                    logger.info(f"Moving {html_file.name} to debug_html/")
                    shutil.move(str(html_file), str(dst))
                    moved_count += 1
        else:
            # Handle specific files
            html_file = public_dir / pattern
            if html_file.exists():
                dst = debug_html_dir / pattern
                logger.info(f"Moving {pattern} to debug_html/")
                shutil.move(str(html_file), str(dst))
                moved_count += 1
                
    logger.info(f"Moved {moved_count} debug HTML files to {debug_html_dir}")
    return moved_count


def consolidate_log_files(project_root: Path, logger):
    """Consolidate scattered log files"""
    logs_dir = project_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Archive old logs
    archive_dir = logs_dir / 'archive' / datetime.now().strftime('%Y%m%d')
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    log_files = [
        'api.log',
        'recursia_runtime.log',
        'osh_validation.log',
        'validation_suite.log',
        'start_backend.log',
        'start_final.log'
    ]
    
    moved_count = 0
    for log_file in log_files:
        src = project_root / log_file
        if src.exists():
            dst = archive_dir / log_file
            logger.info(f"Archiving {log_file} to {archive_dir}")
            shutil.move(str(src), str(dst))
            moved_count += 1
            
    # Also handle frontend logs
    frontend_logs = [
        project_root / 'frontend' / 'dev_server.log',
        project_root / 'frontend' / 'frontend.log'
    ]
    
    for log_path in frontend_logs:
        if log_path.exists():
            dst = archive_dir / log_path.name
            logger.info(f"Archiving {log_path.name} to {archive_dir}")
            shutil.move(str(log_path), str(dst))
            moved_count += 1
            
    logger.info(f"Archived {moved_count} log files to {archive_dir}")
    return moved_count


def clean_old_validation_results(project_root: Path, logger, days_to_keep=7):
    """Clean old validation results and checkpoints"""
    cleaned_count = 0
    
    # Directories to check
    dirs_to_clean = [
        project_root / 'test_results',
        project_root / 'test_validation_output',
        project_root / 'validation_checkpoints',
        project_root / 'validation_results'
    ]
    
    cutoff_date = datetime.now() - timedelta(days=days_to_keep)
    
    for dir_path in dirs_to_clean:
        if not dir_path.exists():
            continue
            
        # Create archive directory
        archive_dir = project_root / 'validation_archive' / dir_path.name
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        # Check each file/subdirectory
        for item in dir_path.iterdir():
            try:
                # Get modification time
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                
                if mtime < cutoff_date:
                    dst = archive_dir / item.name
                    logger.info(f"Archiving old {item.name} from {dir_path.name}")
                    
                    if item.is_file():
                        shutil.move(str(item), str(dst))
                    else:
                        shutil.move(str(item), str(dst))
                    cleaned_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing {item}: {e}")
                
    logger.info(f"Archived {cleaned_count} old validation artifacts")
    return cleaned_count


def create_gitignore_entries(project_root: Path, logger):
    """Add cleanup directories to .gitignore if not already present"""
    gitignore_path = project_root / '.gitignore'
    
    entries_to_add = [
        '\n# Debug and test artifacts',
        'logs/',
        'validation_archive/',
        'tests/debug_moved/',
        'frontend/public/debug_html/',
        '*.log',
        'test_results/',
        'test_validation_output/',
        'validation_checkpoints/',
        'validation_results/'
    ]
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            content = f.read()
            
        # Check which entries need to be added
        new_entries = []
        for entry in entries_to_add:
            if entry not in content and not entry.startswith('\n#'):
                new_entries.append(entry)
                
        if new_entries:
            with open(gitignore_path, 'a') as f:
                f.write('\n')
                for entry in entries_to_add:
                    if entry.startswith('\n#') or entry in new_entries:
                        f.write(f"{entry}\n")
                        
            logger.info(f"Added {len(new_entries)} entries to .gitignore")
    else:
        logger.warning(".gitignore not found")


def main():
    """Run the cleanup process"""
    logger = setup_logging()
    project_root = Path(__file__).parent.parent
    
    logger.info("Starting debug artifact cleanup")
    logger.info(f"Project root: {project_root}")
    
    total_cleaned = 0
    
    # Move test files
    count = move_test_files_to_tests_dir(project_root, logger)
    total_cleaned += count
    
    # Organize debug HTML files
    count = organize_debug_html_files(project_root, logger)
    total_cleaned += count
    
    # Consolidate log files
    count = consolidate_log_files(project_root, logger)
    total_cleaned += count
    
    # Clean old validation results
    count = clean_old_validation_results(project_root, logger)
    total_cleaned += count
    
    # Update .gitignore
    create_gitignore_entries(project_root, logger)
    
    logger.info(f"\nCleanup complete! Organized {total_cleaned} items total.")
    logger.info("All files have been archived, not deleted. You can find them in:")
    logger.info("  - logs/archive/")
    logger.info("  - validation_archive/")
    logger.info("  - tests/debug_moved/")
    logger.info("  - frontend/public/debug_html/")
    

if __name__ == "__main__":
    main()