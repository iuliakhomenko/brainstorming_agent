#!/usr/bin/env python3
"""
Main Text Preprocessor - Refactored into manageable components
AI Brainstorming Agent - Foundation & Knowledge Base - Step 2.1
"""

import logging
from pathlib import Path
from typing import Dict
from data.text_preprocessing.text_cleaner import TextCleaner
from data.text_preprocessing.format_standardizer import FormatStandardizer
from data.text_preprocessing.file_utils import FileUtils


class TextPreprocessor:
    """
    Main text preprocessing pipeline coordinator.
    Handles file I/O and orchestrates the cleaning process.
    """

    def __init__(self, input_dir: str, output_dir: str = None, log_level: str = "INFO"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) if output_dir else self.input_dir / "cleaned"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.cleaner = TextCleaner()
        self.formatter = FormatStandardizer()
        self.file_utils = FileUtils()

    def process_file(self, file_path: Path) -> bool:
        """Process a single text file."""
        try:
            self.logger.info(f"Processing: {file_path.name}")

            # Read file
            text = self.file_utils.read_file(file_path)
            if not text:
                self.logger.warning(f"Empty or unreadable file: {file_path.name}")
                return False

            original_length = len(text)
            self.logger.info(f"Original length: {original_length} chars")

            # Clean text
            text = self.cleaner.clean_text(text)

            # Format text
            text = self.formatter.standardize_format(text)

            final_length = len(text)
            self.logger.info(f"Final length: {final_length} chars ({final_length / original_length * 100:.1f}%)")

            # Save cleaned text
            output_file = self.output_dir / f"cleaned_{file_path.name}"
            self.file_utils.write_file(output_file, text)

            self.logger.info(f"Saved: {output_file.name}")
            return True

        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {e}")
            return False

    def process_directory(self) -> Dict[str, bool]:
        """Process all text files in the input directory."""
        if not self.input_dir.exists():
            self.logger.error(f"Input directory not found: {self.input_dir}")
            return {}

        text_files = list(self.input_dir.glob("*.txt"))
        if not text_files:
            self.logger.warning(f"No .txt files found in {self.input_dir}")
            return {}

        self.logger.info(f"Found {len(text_files)} files to process")

        results = {}
        for file_path in text_files:
            results[file_path.name] = self.process_file(file_path)

        successful = sum(results.values())
        self.logger.info(f"Complete: {successful}/{len(results)} files processed")
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")
    parser.add_argument("input_dir", help="Directory with text files")
    parser.add_argument("-o", "--output_dir", help="Output directory")
    parser.add_argument("-l", "--log_level", choices=["DEBUG", "INFO", "WARNING"],
                        default="INFO", help="Logging level")

    args = parser.parse_args()

    preprocessor = TextPreprocessor(args.input_dir, args.output_dir, args.log_level)
    results = preprocessor.process_directory()

    # Print summary
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    for filename, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {filename}")