#!/usr/bin/env python3
"""
Text Cleaner Module
Handles OCR error correction and character normalization
"""

import re
import unicodedata
import logging


class TextCleaner:
    """Handles text cleaning operations - OCR fixes and character normalization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Common OCR error corrections
        self.ocr_fixes = {
            r'\brn\b': 'm',  # rn -> m
            r'\bvv\b': 'w',  # vv -> w
            r'\btlie\b': 'the',  # tlie -> the
            r'\btliat\b': 'that',  # tliat -> that
            r'\bwliich\b': 'which',  # wliich -> which
            r'\bliave\b': 'have',  # liave -> have
        }

        # Unicode character replacements
        self.unicode_fixes = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '–': '-', '—': '-',  # Em/en dashes
            '…': '...',  # Ellipsis
            '×': 'x', '÷': '/',  # Math symbols
        }

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to decomposed form
        text = unicodedata.normalize('NFD', text)

        # Apply Unicode fixes
        for old, new in self.unicode_fixes.items():
            text = text.replace(old, new)

        # Keep only printable ASCII + basic accents
        cleaned = []
        for char in text:
            if ord(char) < 127 or char in 'áéíóúñüç':
                cleaned.append(char)
            elif char.isspace():
                cleaned.append(' ')
            # Skip other non-ASCII characters

        return ''.join(cleaned)

    def fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR recognition errors."""
        # Apply OCR corrections
        for pattern, replacement in self.ocr_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Space after punctuation

        return text

    def clean_text(self, text: str) -> str:
        """Apply all text cleaning operations."""
        self.logger.debug("Cleaning text...")

        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)

        # Step 2: OCR error fixes
        text = self.fix_ocr_errors(text)

        self.logger.debug("Text cleaning complete")
        return text