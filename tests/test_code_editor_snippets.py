#!/usr/bin/env python3
"""
Test script to validate all quantum code snippets from the code editor can be parsed correctly.
This ensures all snippets in quantumCodeSnippets.json work with the actual Recursia parser.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.direct_parser import DirectParser
except ImportError as e:
    print(f"Error importing parser: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


@dataclass
class TestResult:
    """Result of testing a single snippet"""
    category: str
    snippet_name: str
    success: bool
    error: str = None
    line_count: int = 0
    parse_time_ms: float = 0.0


class SnippetTester:
    """Test all code snippets from the JSON file"""
    
    def __init__(self, json_path: str):
        self.json_path = Path(json_path)
        self.snippets_data = None
        self.results: List[TestResult] = []
        self.parser = DirectParser()
        
    def load_snippets(self) -> bool:
        """Load snippets from JSON file"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.snippets_data = data.get('categories', {})
            return True
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return False
    
    def parse_snippet(self, code: str) -> Tuple[bool, str, float]:
        """
        Parse a single snippet and return success status, error message, and parse time.
        """
        import time
        
        try:
            # Remove any Windows-style line endings
            code = code.replace('\r\n', '\n')
            
            # Measure parse time
            start_time = time.time()
            
            # Parse using DirectParser
            parser = DirectParser()
            ast = parser.parse(code)
            
            end_time = time.time()
            parse_time_ms = (end_time - start_time) * 1000
            
            # If we get here, parsing succeeded
            return True, None, parse_time_ms
            
        except Exception as e:
            # Parsing failed
            error_msg = str(e)
            return False, error_msg, 0.0
    
    def test_all_snippets(self):
        """Test all snippets in all categories"""
        if not self.snippets_data:
            print("No snippets loaded")
            return
        
        total_snippets = 0
        
        # Count total snippets first
        for category_id, category_data in self.snippets_data.items():
            snippets = category_data.get('snippets', [])
            total_snippets += len(snippets)
        
        print(f"Testing {total_snippets} snippets across {len(self.snippets_data)} categories...\n")
        
        # Test each category
        for category_id, category_data in self.snippets_data.items():
            category_title = category_data.get('title', category_id)
            snippets = category_data.get('snippets', [])
            
            print(f"\n{'='*60}")
            print(f"Category: {category_title} ({len(snippets)} snippets)")
            print(f"{'='*60}")
            
            for snippet in snippets:
                snippet_name = snippet.get('name', 'Unnamed')
                snippet_code = snippet.get('code', '')
                
                # Count lines
                line_count = len(snippet_code.split('\n'))
                
                # Test the snippet
                success, error, parse_time = self.parse_snippet(snippet_code)
                
                result = TestResult(
                    category=category_title,
                    snippet_name=snippet_name,
                    success=success,
                    error=error,
                    line_count=line_count,
                    parse_time_ms=parse_time
                )
                self.results.append(result)
                
                # Print result
                if success:
                    print(f"  ✓ {snippet_name} ({line_count} lines, {parse_time:.2f}ms)")
                else:
                    print(f"  ✗ {snippet_name} ({line_count} lines)")
                    print(f"    Error: {error}")
    
    def print_summary(self):
        """Print a summary of test results"""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}\n")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed
        
        print(f"Total snippets tested: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        
        # Average parse time for successful parses
        successful_results = [r for r in self.results if r.success]
        if successful_results:
            avg_parse_time = sum(r.parse_time_ms for r in successful_results) / len(successful_results)
            print(f"Average parse time: {avg_parse_time:.2f}ms")
        
        # Show failures by category
        if failed > 0:
            print(f"\n{'='*60}")
            print("FAILURES BY CATEGORY")
            print(f"{'='*60}\n")
            
            failures_by_category: Dict[str, List[TestResult]] = {}
            for result in self.results:
                if not result.success:
                    if result.category not in failures_by_category:
                        failures_by_category[result.category] = []
                    failures_by_category[result.category].append(result)
            
            for category, failures in failures_by_category.items():
                print(f"\n{category}: {len(failures)} failures")
                for failure in failures:
                    print(f"  - {failure.snippet_name}")
                    print(f"    {failure.error}")
        
        # Show category statistics
        print(f"\n{'='*60}")
        print("CATEGORY STATISTICS")
        print(f"{'='*60}\n")
        
        category_stats: Dict[str, Dict[str, int]] = {}
        for result in self.results:
            if result.category not in category_stats:
                category_stats[result.category] = {'total': 0, 'passed': 0}
            category_stats[result.category]['total'] += 1
            if result.success:
                category_stats[result.category]['passed'] += 1
        
        for category, stats in category_stats.items():
            total = stats['total']
            passed = stats['passed']
            print(f"{category}: {passed}/{total} passed ({passed/total*100:.1f}%)")
    
    def save_detailed_report(self, output_path: str = None):
        """Save a detailed report to a file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"snippet_test_report_{timestamp}.txt"
        
        with open(output_path, 'w') as f:
            f.write("Recursia Code Snippet Test Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"JSON File: {self.json_path}\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            total = len(self.results)
            passed = sum(1 for r in self.results if r.success)
            failed = total - passed
            
            f.write("SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total snippets: {total}\n")
            f.write(f"Passed: {passed} ({passed/total*100:.1f}%)\n")
            f.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS\n")
            f.write("-"*40 + "\n\n")
            
            current_category = None
            for result in self.results:
                if result.category != current_category:
                    current_category = result.category
                    f.write(f"\n[{current_category}]\n")
                
                status = "PASS" if result.success else "FAIL"
                f.write(f"  {status}: {result.snippet_name} ({result.line_count} lines")
                if result.success:
                    f.write(f", {result.parse_time_ms:.2f}ms)\n")
                else:
                    f.write(")\n")
                    f.write(f"    Error: {result.error}\n")
            
            # Failed snippets detail
            if failed > 0:
                f.write("\n\nFAILED SNIPPETS DETAIL\n")
                f.write("-"*40 + "\n\n")
                
                for result in self.results:
                    if not result.success:
                        f.write(f"Category: {result.category}\n")
                        f.write(f"Snippet: {result.snippet_name}\n")
                        f.write(f"Error: {result.error}\n")
                        f.write("-"*20 + "\n\n")
        
        print(f"\nDetailed report saved to: {output_path}")


def main():
    """Main entry point"""
    # Path to the JSON file
    json_path = Path(__file__).parent.parent / "frontend" / "src" / "data" / "quantumCodeSnippets.json"
    
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        sys.exit(1)
    
    print(f"Loading snippets from: {json_path}")
    
    # Create tester
    tester = SnippetTester(json_path)
    
    # Load snippets
    if not tester.load_snippets():
        sys.exit(1)
    
    # Test all snippets
    tester.test_all_snippets()
    
    # Print summary
    tester.print_summary()
    
    # Save detailed report
    tester.save_detailed_report()
    
    # Exit with appropriate code
    failed_count = sum(1 for r in tester.results if not r.success)
    sys.exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()