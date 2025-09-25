#!/usr/bin/env python3
"""Build Lambda deployment package."""

import shutil
import subprocess
import sys
from pathlib import Path


def build_lambda():
    """Build Lambda deployment package."""
    print("Building Lambda deployment package...")

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
        if item.name != "__pycache__":
            if item.is_dir():
                shutil.copytree(item, build_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, build_dir / item.name)

    # Copy core package
    print("Copying core package...")
    core_src = project_root / "core"
    shutil.copytree(core_src, build_dir / "core", dirs_exist_ok=True)

    # Install dependencies
    print("Installing dependencies...")
    requirements_file = build_dir / "requirements.txt"
    if requirements_file.exists():
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_file),
                "-t",
                str(build_dir),
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False

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

    # Clean up build directory
    shutil.rmtree(build_dir)
    print("✓ Cleaned up build directory")

    return True


if __name__ == "__main__":
    success = build_lambda()
    sys.exit(0 if success else 1)
