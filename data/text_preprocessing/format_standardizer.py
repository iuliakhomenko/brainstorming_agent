#!/usr/bin/env python3
"""
Format Standardizer Module
Handles text formatting and structure standardization
"""

import re
import logging
from typing import List


class FormatStandardizer:
    """Handles text formatting standardization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Patterns for headers/footers to remove (conservative patterns)
        self.removal_patterns = [
            r'^\s*\d{1,3}\s*$',  # Page numbers (1-3 digits)
            r'^\s*Page\s+\d+\s*$',  # "Page X"
            r'^\s*Chapter\s+\d+\s*$',  # "Chapter X"
            r'^\s*©\s*\d{4}.*$',  # Copyright lines
            r'^\s*ISBN[:\s-]*[\d-]+\s*$',  # ISBN numbers
        ]

    def remove_headers_footers(self, text: str) -> str:
        """Remove headers, footers, and page numbers."""
        lines = text.split('\n')
        cleaned_lines = []
        removed_count = 0

        for line in lines:
            original_line = line
            stripped = line.strip()

            # Keep empty lines and very short lines
            if len(stripped) < 3:
                cleaned_lines.append(original_line)
                continue

            # Check if line should be removed
            should_remove = False
            for pattern in self.removal_patterns:
                if re.match(pattern, stripped, re.IGNORECASE):
                    should_remove = True
                    removed_count += 1
                    self.logger.debug(f"Removing: {stripped[:30]}...")
                    break

            if not should_remove:
                cleaned_lines.append(original_line)

        self.logger.info(f"Removed {removed_count} header/footer lines")
        return '\n'.join(cleaned_lines)

    def fix_line_breaks(self, text: str) -> str:
        """Fix paragraph breaks and line spacing."""
        # Fix hyphenated words split across lines
        text = re.sub(r'-\s*\n\s*([a-z])', r'\1', text)

        # Standardize paragraph breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple breaks -> double

        # Clean up line whitespace
        text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Leading whitespace
        text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)  # Trailing whitespace

        return text

    def standardize_punctuation(self, text: str) -> str:
        """Standardize punctuation and quotes."""
        # Fix quotes and apostrophes
        text = re.sub(r'[`´'']', "'", text)
        text = re.sub(r'["""]', '"', text)

        # Fix multiple punctuation
        text = re.sub(r'\.{2,}', '...', text)  # Multiple dots -> ellipsis
        text = re.sub(r' +', ' ', text)  # Multiple spaces -> single

        return text.strip()

    def standardize_format(self, text: str) -> str:
        """Apply all formatting standardization."""
        self.logger.debug("Standardizing format...")

        # Step 1: Remove headers/footers
        text = self.remove_headers_footers(text)

        # Step 2: Fix line breaks
        text = self.fix_line_breaks(text)

        # Step 3: Standardize punctuation
        text = self.standardize_punctuation(text)

        self.logger.debug("Format standardization complete")
        return text