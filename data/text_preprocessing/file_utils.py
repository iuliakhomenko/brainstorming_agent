#!/usr/bin/env python3
"""
File Utilities Module
Handles file reading/writing with encoding detection
"""

import chardet
import logging
from pathlib import Path


class FileUtils:
    """Handles file I/O operations with encoding detection."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB for detection
                result = chardet.detect(raw_data)

            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)

            self.logger.debug(f"Encoding for {file_path.name}: {encoding} (confidence: {confidence:.2f})")

            # Use utf-8 as fallback if confidence is low
            if confidence < 0.7:
                self.logger.warning(f"Low confidence encoding detection, using utf-8")
                return 'utf-8'

            return encoding

        except Exception as e:
            self.logger.error(f"Encoding detection failed for {file_path.name}: {e}")
            return 'utf-8'

    def read_file(self, file_path: Path) -> str:
        """Read text file with automatic encoding detection."""
        try:
            encoding = self.detect_encoding(file_path)

            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()

            self.logger.debug(f"Successfully read {file_path.name} ({len(content)} chars)")
            return content

        except Exception as e:
            self.logger.error(f"Failed to read {file_path.name}: {e}")
            return ""

    def write_file(self, file_path: Path, content: str) -> bool:
        """Write text file in UTF-8 encoding."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            self.logger.debug(f"Successfully wrote {file_path.name} ({len(content)} chars)")
            return True

        except Exception as e:
            self.logger.error(f"Failed to write {file_path.name}: {e}")
            return False