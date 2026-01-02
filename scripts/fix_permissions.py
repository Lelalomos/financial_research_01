#!/usr/bin/env python
"""
Fix permissions for files owned by root in the data directory.

This script attempts to fix file permissions that were created by Docker
containers running as root. It requires appropriate permissions to modify
the files.
"""

import os
import subprocess
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import get_logger

logger = get_logger("fix_permissions", log_dir="logs")


def find_root_owned_files(base_dir: Path) -> list[Path]:
    """Find files owned by root."""
    root_files = []
    for item in base_dir.rglob("*"):
        if item.is_file():
            try:
                stat_info = os.stat(item)
                # Check if file is owned by root (uid 0)
                if stat_info.st_uid == 0:
                    root_files.append(item)
            except (OSError, FileNotFoundError):
                continue
    return root_files


def fix_with_chmod(file_path: Path) -> bool:
    """Try to fix permissions using chmod u+w."""
    try:
        # Add user write permission
        os.chmod(file_path, 0o644)
        logger.info(f"Fixed permissions with chmod: {file_path}")
        return True
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not chmod {file_path}: {e}")
        return False


def fix_with_chown(file_path: Path) -> bool:
    """Try to fix ownership using chown."""
    try:
        # Get current user
        import pwd
        uid = pwd.getpwuid(os.getuid()).pw_uid
        gid = pwd.getpwuid(os.getuid()).pw_gid
        os.chown(file_path, uid, gid)
        logger.info(f"Fixed ownership with chown: {file_path}")
        return True
    except (OSError, PermissionError) as e:
        logger.warning(f"Could not chown {file_path}: {e}")
        return False


def delete_and_recreate(root_files: list[Path]) -> bool:
    """
    Delete root-owned files and suggest re-running preprocessing.
    This is the safest option when permission changes fail.
    """
    print("\n" + "=" * 60)
    print("ROOT-OWNED FILES DETECTED")
    print("=" * 60)
    print(f"\nFound {len(root_files)} files owned by root.")
    print("\nThese files were created by Docker containers running as root.")
    print("You have two options:")
    print("\n1. Delete the files and re-run preprocessing (recommended)")
    print("2. Use sudo to change ownership")

    choice = input("\nDelete root-owned files? (y/n): ").strip().lower()

    if choice == 'y':
        deleted = 0
        for file_path in root_files:
            try:
                os.remove(file_path)
                logger.info(f"Deleted: {file_path}")
                deleted += 1
            except (OSError, PermissionError) as e:
                logger.error(f"Could not delete {file_path}: {e}")

        # Also remove empty directories
        data_dir = Path(__file__).parent.parent / "data"
        for dir_path in [data_dir / "processed" / "train",
                         data_dir / "processed" / "val",
                         data_dir / "processed" / "test"]:
            if dir_path.exists() and not list(dir_path.iterdir()):
                try:
                    os.rmdir(dir_path)
                    logger.info(f"Removed empty directory: {dir_path}")
                except OSError:
                    pass

        print(f"\nDeleted {deleted} files.")
        print("\nPlease re-run preprocessing:")
        print("  python scripts/preprocess_data.py --output-dir data/processed")
        return True
    else:
        print("\nTo fix manually, run:")
        print(f"  sudo chown -R $USER:$USER {data_dir}")
        return False


def main():
    """Main function to fix permissions."""
    data_dir = Path(__file__).parent.parent / "data"

    logger.info("=" * 60)
    logger.info("FIXING PERMISSIONS")
    logger.info("=" * 60)
    logger.info(f"Scanning: {data_dir}")

    root_files = find_root_owned_files(data_dir)

    if not root_files:
        logger.info("No root-owned files found. All good!")
        return 0

    logger.warning(f"Found {len(root_files)} root-owned files:")
    for f in root_files[:10]:
        logger.warning(f"  {f}")
    if len(root_files) > 10:
        logger.warning(f"  ... and {len(root_files) - 10} more")

    # Try to fix permissions
    fixed_chmod = 0
    for file_path in root_files:
        if fix_with_chmod(file_path):
            fixed_chmod += 1

    if fixed_chmod == len(root_files):
        logger.info(f"Successfully fixed all {fixed_chmod} files with chmod")
        return 0

    # Try chown for remaining files
    remaining = [f for f in root_files if not os.access(f, os.W_OK)]
    fixed_chown = 0
    for file_path in remaining:
        if fix_with_chown(file_path):
            fixed_chown += 1

    if fixed_chown == len(remaining):
        logger.info(f"Successfully fixed all remaining {fixed_chown} files with chown")
        return 0

    # If still can't fix, offer to delete
    still_root = [f for f in root_files if not os.access(f, os.W_OK)]
    if still_root:
        if delete_and_recreate(still_root):
            return 0

    logger.error("Could not fix all permissions. Please run with sudo:")
    logger.error(f"  sudo {sys.argv[0]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
