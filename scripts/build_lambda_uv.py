#!/usr/bin/env python3
"""Build Lambda deployment package using uv."""

import shutil
import subprocess
import sys
from pathlib import Path


def build_lambda():
    """Build Lambda deployment package with uv."""
    print("Building Lambda deployment package with uv...")

    # Get project root
    project_root = Path(__file__).parent.parent

    # Clean up previous build
    build_dir = project_root / "build" / "lambda"
    zip_file = project_root / "lambda-deployment.zip"

    if build_dir.exists():
        shutil.rmtree(build_dir)
    if zip_file.exists():
        zip_file.unlink()

    # Create build directory
    build_dir.mkdir(parents=True, exist_ok=True)
    print("✓ Created build directory")

    # Copy Lambda code
    print("Copying Lambda code...")
    lambda_src = project_root / "lambda"
    for item in lambda_src.iterdir():
        if item.name not in ("__pycache__", "requirements.txt"):
            if item.is_dir():
                shutil.copytree(item, build_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, build_dir / item.name)

    # Copy core package
    print("Copying core package...")
    core_src = project_root / "core"
    shutil.copytree(core_src, build_dir / "core", dirs_exist_ok=True)

    # Install dependencies with uv
    print("Installing dependencies with uv...")
    requirements_file = lambda_src / "requirements.txt"
    if requirements_file.exists():
        # Use uv pip install to target directory
        result = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-r",
                str(requirements_file),
                "--target",
                str(build_dir),
                "--python",
                "3.11",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False

    # Clean up unnecessary files
    print("Cleaning up unnecessary files...")
    # Remove test directories
    for test_dir in build_dir.rglob("tests"):
        if test_dir.is_dir():
            shutil.rmtree(test_dir)
    # Remove __pycache__ directories
    for cache_dir in build_dir.rglob("__pycache__"):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    # Remove .pyc files
    for pyc_file in build_dir.rglob("*.pyc"):
        pyc_file.unlink()
    # Remove .dist-info directories (keep only essential)
    for dist_dir in build_dir.glob("*.dist-info"):
        if dist_dir.is_dir():
            shutil.rmtree(dist_dir)

    # Create deployment package
    print("Creating deployment package...")
    shutil.make_archive(str(zip_file.with_suffix("")), "zip", build_dir)

    # Check package size
    package_size_bytes = zip_file.stat().st_size
    package_size_mb = package_size_bytes / (1024 * 1024)

    print(f"✓ Lambda package created: {zip_file.name}")
    print(f"  Package size: {package_size_mb:.2f} MB")

    if package_size_mb > 50:
        print("⚠️  Warning: Package size exceeds 50MB (Lambda unzipped limit is 250MB)")
    elif package_size_mb < 20:
        print("✓ Package size is optimal for Lambda cold starts")

    # Clean up build directory
    shutil.rmtree(build_dir)
    print("✓ Cleaned up build directory")

    return True


if __name__ == "__main__":
    success = build_lambda()
    sys.exit(0 if success else 1)
