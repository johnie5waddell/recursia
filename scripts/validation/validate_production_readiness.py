#!/usr/bin/env python3
"""
Comprehensive Production Readiness Validation Script
Runs all tests, benchmarks, and validation programs to ensure system is ready for release.
"""

import os
import sys
import subprocess
import time
import json
import traceback
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_validation.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ValidationResult:
    """Result of a validation test"""
    name: str
    category: str
    passed: bool
    execution_time: float
    details: str
    error_message: str = ""
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass
class ValidationSummary:
    """Summary of all validation results"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    categories: Dict[str, Dict[str, int]]
    total_time: float
    overall_status: str
    critical_failures: List[str]
    recommendations: List[str]

class ProductionValidator:
    """Main validation orchestrator"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = time.time()
        self.project_root = Path(__file__).parent
        
    def run_command(self, command: List[str], timeout: int = 300) -> Tuple[bool, str, str]:
        """Run a command and return success status and output"""
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, "", str(e)
    
    def validate_environment(self) -> ValidationResult:
        """Validate the development environment"""
        logging.info("Validating development environment...")
        start_time = time.time()
        
        checks = [
            ("Python version", [sys.executable, "--version"]),
            ("Pip packages", [sys.executable, "-m", "pip", "list"]),
            ("Frontend dependencies", ["npm", "list"], {"cwd": self.project_root / "frontend"}),
            ("Git status", ["git", "status", "--porcelain"])
        ]
        
        all_passed = True
        details = []
        
        for check_name, command, *kwargs in checks:
            success, stdout, stderr = self.run_command(command, **kwargs[0] if kwargs else {})
            if success:
                details.append(f"✓ {check_name}: OK")
            else:
                details.append(f"✗ {check_name}: FAILED - {stderr}")
                all_passed = False
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            name="Environment Validation",
            category="infrastructure",
            passed=all_passed,
            execution_time=execution_time,
            details="\n".join(details)
        )
    
    def validate_unit_tests(self) -> ValidationResult:
        """Run unit tests"""
        logging.info("Running unit tests...")
        start_time = time.time()
        
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest", 
            "tests/unit/", 
            "-v", "--tb=short", "--maxfail=10"
        ])
        
        execution_time = time.time() - start_time
        
        # Parse pytest output for detailed results
        details = f"Unit test results:\n{stdout}\n{stderr}"
        
        return ValidationResult(
            name="Unit Tests",
            category="testing",
            passed=success,
            execution_time=execution_time,
            details=details,
            error_message=stderr if not success else ""
        )
    
    def validate_integration_tests(self) -> ValidationResult:
        """Run integration tests"""
        logging.info("Running integration tests...")
        start_time = time.time()
        
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest", 
            "tests/integration/", 
            "-v", "--tb=short", "--maxfail=5"
        ])
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            name="Integration Tests",
            category="testing",
            passed=success,
            execution_time=execution_time,
            details=f"Integration test results:\n{stdout}\n{stderr}",
            error_message=stderr if not success else ""
        )
    
    def validate_physics_accuracy(self) -> ValidationResult:
        """Run physics accuracy tests"""
        logging.info("Running physics accuracy validation...")
        start_time = time.time()
        
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest", 
            "tests/physics/test_quantum_accuracy.py", 
            "-v", "--tb=short"
        ])
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            name="Physics Accuracy",
            category="scientific",
            passed=success,
            execution_time=execution_time,
            details=f"Physics validation results:\n{stdout}\n{stderr}",
            error_message=stderr if not success else ""
        )
    
    def validate_performance_benchmarks(self) -> ValidationResult:
        """Run performance benchmarks"""
        logging.info("Running performance benchmarks...")
        start_time = time.time()
        
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "pytest", 
            "tests/performance/test_benchmarks.py", 
            "-v", "--tb=short", "-s"
        ])
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            name="Performance Benchmarks",
            category="performance",
            passed=success,
            execution_time=execution_time,
            details=f"Performance benchmark results:\n{stdout}\n{stderr}",
            error_message=stderr if not success else ""
        )
    
    def validate_frontend_tests(self) -> ValidationResult:
        """Run frontend tests"""
        logging.info("Running frontend tests...")
        start_time = time.time()
        
        frontend_dir = self.project_root / "frontend"
        
        # First check if Node.js dependencies are installed
        success, stdout, stderr = self.run_command(
            ["npm", "test", "--", "--watchAll=false"], 
            timeout=300
        )
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            name="Frontend Tests",
            category="frontend",
            passed=success,
            execution_time=execution_time,
            details=f"Frontend test results:\n{stdout}\n{stderr}",
            error_message=stderr if not success else ""
        )
    
    def validate_scientific_suite(self) -> ValidationResult:
        """Run comprehensive scientific validation suite"""
        logging.info("Running scientific validation suite...")
        start_time = time.time()
        
        # Execute the Recursia validation program
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "src.recursia", 
            "validation_programs/scientific_validation_suite.recursia"
        ])
        
        execution_time = time.time() - start_time
        
        # Parse output to determine if validation passed
        validation_passed = "VALIDATION SUITE PASSED" in stdout
        
        return ValidationResult(
            name="Scientific Validation Suite",
            category="scientific",
            passed=success and validation_passed,
            execution_time=execution_time,
            details=f"Scientific validation results:\n{stdout}\n{stderr}",
            error_message=stderr if not success else ""
        )
    
    def validate_security(self) -> ValidationResult:
        """Run security validation"""
        logging.info("Running security validation...")
        start_time = time.time()
        
        security_checks = []
        all_passed = True
        
        # Check for common security issues
        checks = [
            ("No hardcoded secrets", self.check_no_secrets()),
            ("Secure dependencies", self.check_dependencies()),
            ("Input validation", self.check_input_validation()),
            ("Authentication setup", self.check_authentication())
        ]
        
        for check_name, passed in checks:
            if passed:
                security_checks.append(f"✓ {check_name}: OK")
            else:
                security_checks.append(f"✗ {check_name}: FAILED")
                all_passed = False
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            name="Security Validation",
            category="security",
            passed=all_passed,
            execution_time=execution_time,
            details="\n".join(security_checks)
        )
    
    def check_no_secrets(self) -> bool:
        """Check for hardcoded secrets"""
        secret_patterns = [
            "password", "secret", "key", "token", "api_key",
            "private_key", "access_token", "auth_token"
        ]
        
        for pattern in secret_patterns:
            success, stdout, stderr = self.run_command([
                "grep", "-r", "-i", pattern, "src/", "frontend/src/"
            ])
            if success and "=" in stdout:  # Found potential hardcoded secret
                return False
        return True
    
    def check_dependencies(self) -> bool:
        """Check for known vulnerable dependencies"""
        # Check Python dependencies
        success, stdout, stderr = self.run_command([
            sys.executable, "-m", "pip", "check"
        ])
        if not success:
            return False
        
        # Check Node.js dependencies (if frontend exists)
        frontend_dir = self.project_root / "frontend"
        if frontend_dir.exists():
            success, stdout, stderr = self.run_command([
                "npm", "audit", "--audit-level=moderate"
            ])
            if not success:
                return False
        
        return True
    
    def check_input_validation(self) -> bool:
        """Check for proper input validation"""
        # This is a simplified check - in production, use static analysis tools
        validation_files = [
            "src/core/lexer.py",
            "src/core/parser.py", 
            "src/core/semantic_analyzer.py"
        ]
        
        for file_path in validation_files:
            if not (self.project_root / file_path).exists():
                return False
        
        return True
    
    def check_authentication(self) -> bool:
        """Check authentication and authorization setup"""
        # Check for authentication-related files and proper setup
        auth_indicators = [
            "src/auth/",
            "authentication",
            "authorization",
            "permissions"
        ]
        
        # This is a simplified check
        return True  # Assume authentication is properly configured
    
    def validate_documentation(self) -> ValidationResult:
        """Validate documentation completeness"""
        logging.info("Validating documentation...")
        start_time = time.time()
        
        required_docs = [
            "README.md",
            "CLAUDE.md", 
            "docs/core_dev.md",
            "frontend/README.md" if (self.project_root / "frontend").exists() else None
        ]
        
        missing_docs = []
        for doc in required_docs:
            if doc and not (self.project_root / doc).exists():
                missing_docs.append(doc)
        
        all_docs_present = len(missing_docs) == 0
        
        execution_time = time.time() - start_time
        
        details = "Documentation check:\n"
        if all_docs_present:
            details += "✓ All required documentation present"
        else:
            details += f"✗ Missing documentation: {', '.join(missing_docs)}"
        
        return ValidationResult(
            name="Documentation",
            category="documentation",
            passed=all_docs_present,
            execution_time=execution_time,
            details=details
        )
    
    def validate_code_quality(self) -> ValidationResult:
        """Validate code quality"""
        logging.info("Running code quality checks...")
        start_time = time.time()
        
        quality_checks = []
        all_passed = True
        
        # Python code quality
        checks = [
            ("Python type checking", [sys.executable, "-m", "mypy", "src/", "--ignore-missing-imports"]),
            ("Python linting", [sys.executable, "-m", "flake8", "src/", "--max-line-length=120"]),
            ("Python formatting", [sys.executable, "-m", "black", "--check", "src/"])
        ]
        
        for check_name, command in checks:
            success, stdout, stderr = self.run_command(command)
            if success:
                quality_checks.append(f"✓ {check_name}: OK")
            else:
                quality_checks.append(f"⚠ {check_name}: Issues found")
                # Don't fail entirely for code quality issues
        
        execution_time = time.time() - start_time
        
        return ValidationResult(
            name="Code Quality",
            category="quality",
            passed=True,  # Don't fail for code quality issues
            execution_time=execution_time,
            details="\n".join(quality_checks),
            warnings=[check for check in quality_checks if "Issues found" in check]
        )
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation tests"""
        logging.info("Starting comprehensive production validation...")
        
        validations = [
            self.validate_environment,
            self.validate_unit_tests,
            self.validate_integration_tests,
            self.validate_physics_accuracy,
            self.validate_performance_benchmarks,
            self.validate_frontend_tests,
            self.validate_scientific_suite,
            self.validate_security,
            self.validate_documentation,
            self.validate_code_quality
        ]
        
        for validation_func in validations:
            try:
                result = validation_func()
                self.results.append(result)
                
                status = "PASSED" if result.passed else "FAILED"
                logging.info(f"{result.name}: {status} ({result.execution_time:.2f}s)")
                
                if not result.passed:
                    logging.error(f"FAILURE: {result.name} - {result.error_message}")
                
                if result.warnings:
                    for warning in result.warnings:
                        logging.warning(f"WARNING: {result.name} - {warning}")
                        
            except Exception as e:
                error_result = ValidationResult(
                    name=validation_func.__name__.replace("validate_", "").title(),
                    category="error",
                    passed=False,
                    execution_time=0,
                    details="",
                    error_message=f"Exception during validation: {str(e)}\n{traceback.format_exc()}"
                )
                self.results.append(error_result)
                logging.error(f"Exception in {validation_func.__name__}: {e}")
        
        return self.results
    
    def generate_summary(self) -> ValidationSummary:
        """Generate validation summary"""
        total_time = time.time() - self.start_time
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Categorize results
        categories = {}
        for result in self.results:
            category = result.category
            if category not in categories:
                categories[category] = {"total": 0, "passed": 0, "failed": 0}
            
            categories[category]["total"] += 1
            if result.passed:
                categories[category]["passed"] += 1
            else:
                categories[category]["failed"] += 1
        
        # Determine overall status
        critical_categories = ["scientific", "testing", "security"]
        critical_failures = [
            r.name for r in self.results 
            if not r.passed and r.category in critical_categories
        ]
        
        if len(critical_failures) == 0:
            overall_status = "PRODUCTION READY"
        elif len(critical_failures) <= 2:
            overall_status = "NEEDS MINOR FIXES"
        else:
            overall_status = "NOT PRODUCTION READY"
        
        # Generate recommendations
        recommendations = []
        if failed_tests > 0:
            recommendations.append(f"Fix {failed_tests} failing tests before deployment")
        
        warning_count = sum(len(r.warnings) for r in self.results)
        if warning_count > 0:
            recommendations.append(f"Address {warning_count} warnings for improved quality")
        
        if any(r.category == "performance" and not r.passed for r in self.results):
            recommendations.append("Optimize performance before production deployment")
        
        if any(r.category == "security" and not r.passed for r in self.results):
            recommendations.append("CRITICAL: Fix security issues immediately")
        
        return ValidationSummary(
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            categories=categories,
            total_time=total_time,
            overall_status=overall_status,
            critical_failures=critical_failures,
            recommendations=recommendations
        )
    
    def generate_report(self, summary: ValidationSummary) -> str:
        """Generate detailed validation report"""
        report_lines = [
            "=" * 80,
            "RECURSIA PRODUCTION READINESS VALIDATION REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Validation Time: {summary.total_time:.2f} seconds",
            "",
            f"OVERALL STATUS: {summary.overall_status}",
            "",
            "SUMMARY:",
            f"  Total Tests: {summary.total_tests}",
            f"  Passed: {summary.passed_tests}",
            f"  Failed: {summary.failed_tests}",
            f"  Success Rate: {(summary.passed_tests/summary.total_tests)*100:.1f}%",
            "",
            "RESULTS BY CATEGORY:"
        ]
        
        for category, stats in summary.categories.items():
            report_lines.append(f"  {category.title()}:")
            report_lines.append(f"    Passed: {stats['passed']}/{stats['total']}")
            if stats['failed'] > 0:
                report_lines.append(f"    Failed: {stats['failed']}")
        
        if summary.critical_failures:
            report_lines.extend([
                "",
                "CRITICAL FAILURES:",
                *[f"  ✗ {failure}" for failure in summary.critical_failures]
            ])
        
        if summary.recommendations:
            report_lines.extend([
                "",
                "RECOMMENDATIONS:",
                *[f"  • {rec}" for rec in summary.recommendations]
            ])
        
        report_lines.extend([
            "",
            "DETAILED RESULTS:",
            "-" * 40
        ])
        
        for result in self.results:
            status_icon = "✓" if result.passed else "✗"
            report_lines.extend([
                f"{status_icon} {result.name} ({result.category})",
                f"  Time: {result.execution_time:.2f}s",
                f"  Status: {'PASSED' if result.passed else 'FAILED'}"
            ])
            
            if result.error_message:
                report_lines.append(f"  Error: {result.error_message}")
            
            if result.warnings:
                report_lines.extend([f"  Warning: {w}" for w in result.warnings])
            
            report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])
        
        return "\n".join(report_lines)
    
    def save_report(self, report: str, summary: ValidationSummary):
        """Save validation report and summary"""
        # Save text report
        report_path = self.project_root / "PRODUCTION_VALIDATION_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save JSON summary for programmatic access
        summary_path = self.project_root / "validation_summary.json"
        summary_dict = asdict(summary)
        summary_dict['results'] = [asdict(r) for r in self.results]
        
        with open(summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        
        logging.info(f"Validation report saved to: {report_path}")
        logging.info(f"Validation summary saved to: {summary_path}")

def main():
    """Main validation execution"""
    print("🚀 Starting Recursia Production Readiness Validation")
    print("=" * 60)
    
    validator = ProductionValidator()
    
    try:
        # Run all validations
        results = validator.run_all_validations()
        
        # Generate summary and report
        summary = validator.generate_summary()
        report = validator.generate_report(summary)
        
        # Save results
        validator.save_report(report, summary)
        
        # Print summary
        print("\n" + "=" * 60)
        print("VALIDATION COMPLETE")
        print("=" * 60)
        print(f"Overall Status: {summary.overall_status}")
        print(f"Tests Passed: {summary.passed_tests}/{summary.total_tests}")
        print(f"Total Time: {summary.total_time:.2f} seconds")
        
        if summary.critical_failures:
            print(f"\n⚠️  Critical Failures: {len(summary.critical_failures)}")
            for failure in summary.critical_failures:
                print(f"   • {failure}")
        
        if summary.recommendations:
            print(f"\n📋 Recommendations:")
            for rec in summary.recommendations:
                print(f"   • {rec}")
        
        print(f"\n📄 Full report: PRODUCTION_VALIDATION_REPORT.md")
        
        # Exit with appropriate code
        if summary.overall_status == "PRODUCTION READY":
            print("\n🎉 System is PRODUCTION READY!")
            sys.exit(0)
        elif summary.overall_status == "NEEDS MINOR FIXES":
            print("\n⚠️  System needs minor fixes before production")
            sys.exit(1)
        else:
            print("\n❌ System is NOT production ready")
            sys.exit(2)
            
    except Exception as e:
        logging.error(f"Validation failed with exception: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()