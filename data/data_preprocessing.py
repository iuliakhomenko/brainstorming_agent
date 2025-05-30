"""
PDF and EPUB Text Extraction Module for Brainstorming AI System
Implements dual extraction approach for both PDF and EPUB files
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re
import PyPDF2
import pdfplumber
import ebooklib
from ebooklib import epub
import epub2txt
from bs4 import BeautifulSoup
from dataclasses import dataclass
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Container for document extraction results"""
    filename: str
    success: bool
    text: str
    page_count: int
    extraction_method: str
    file_type: str
    error_message: Optional[str] = None
    word_count: int = 0
    processing_time: float = 0.0


class DocumentExtractor:
    """PDF and EPUB text extraction with dual approach and quality validation"""

    def __init__(self, output_dir: str = "data/books"):
        """
        Initialize document extractor

        Args:
            output_dir: Directory to save processed text files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extraction statistics
        self.extraction_stats = {
            'total_files': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'pdf_files': 0,
            'epub_files': 0,
            'pypdf2_success': 0,
            'pdfplumber_success': 0,
            'ebooklib_success': 0,
            'epub2txt_success': 0,
            'both_failed': 0
        }

    # PDF extraction methods (unchanged from original)
    def extract_with_pypdf2(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Extract text using PyPDF2

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, success_flag)
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""

                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1} with PyPDF2: {e}")
                        continue

                return text, len(text.strip()) > 100  # Success if we got substantial text

        except Exception as e:
            logger.error(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return "", False

    def extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, bool]:
        """
        Extract text using pdfplumber (fallback method)

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (extracted_text, success_flag)
        """
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""

                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"

                        # Also try to extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            text += "\n--- Tables ---\n"
                            for table in tables:
                                for row in table:
                                    text += " | ".join([cell or "" for cell in row]) + "\n"
                                text += "\n"

                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1} with pdfplumber: {e}")
                        continue

                return text, len(text.strip()) > 100

        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return "", False

    # EPUB extraction methods (new functionality)
    def extract_with_ebooklib(self, epub_path: str) -> Tuple[str, bool]:
        """
        Extract text using ebooklib (primary EPUB method)

        Args:
            epub_path: Path to EPUB file

        Returns:
            Tuple of (extracted_text, success_flag)
        """
        try:
            book = epub.read_epub(epub_path)
            text = ""
            chapter_count = 0

            # Extract metadata
            title = book.get_metadata('DC', 'title')
            author = book.get_metadata('DC', 'creator')

            if title:
                text += f"Title: {title[0][0]}\n"
            if author:
                text += f"Author: {author[0][0]}\n"
            text += "\n" + "=" * 50 + "\n\n"

            # Extract text from each document/chapter
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    chapter_count += 1
                    try:
                        # Parse HTML content
                        soup = BeautifulSoup(item.get_content(), 'html.parser')

                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()

                        # Extract text
                        chapter_text = soup.get_text()

                        # Clean up whitespace
                        lines = (line.strip() for line in chapter_text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        chapter_text = ' '.join(chunk for chunk in chunks if chunk)

                        if chapter_text.strip():
                            text += f"\n--- Chapter {chapter_count} ---\n"
                            text += chapter_text + "\n\n"

                    except Exception as e:
                        logger.warning(f"Failed to extract chapter {chapter_count} with ebooklib: {e}")
                        continue

            return text, len(text.strip()) > 200  # Success if we got substantial text

        except Exception as e:
            logger.error(f"ebooklib extraction failed for {epub_path}: {e}")
            return "", False

    def extract_with_epub2txt(self, epub_path: str) -> Tuple[str, bool]:
        """
        Extract text using epub2txt (fallback EPUB method)

        Args:
            epub_path: Path to EPUB file

        Returns:
            Tuple of (extracted_text, success_flag)
        """
        try:
            # Extract using epub2txt
            text = epub2txt.epub2txt(epub_path)

            if text and len(text.strip()) > 200:
                # Add some basic structure markers
                formatted_text = "--- EPUB Content ---\n\n" + text
                return formatted_text, True
            else:
                return "", False

        except Exception as e:
            logger.error(f"epub2txt extraction failed for {epub_path}: {e}")
            return "", False

    def clean_extracted_text(self, text: str) -> str:
        """
        Clean and normalize extracted text (enhanced for both PDF and EPUB)

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove excessive whitespace and normalize line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        # Remove common document artifacts
        text = re.sub(r'(?i)(?:page \d+|\d+ page)', '', text)
        text = re.sub(r'\.{3,}', '...', text)  # Multiple dots
        text = re.sub(r'-{2,}', '--', text)  # Multiple dashes

        # Fix common encoding issues
        replacements = {
            '"': '"', '"': '"',  # Smart quotes
            ''': "'", ''': "'",  # Smart apostrophes
            '–': '-', '—': '--',  # Em/en dashes
            '…': '...',  # Ellipsis
            '\uf0b7': '•',  # Bullet point
            '\xa0': ' ',  # Non-breaking space
            '\u00a0': ' ',  # Another non-breaking space
            '\u2028': '\n',  # Line separator
            '\u2029': '\n\n'  # Paragraph separator
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        # Remove structural markers we added (they're for processing only)
        text = re.sub(r'\n--- Page \d+ ---\n', '\n', text)
        text = re.sub(r'\n--- Chapter \d+ ---\n', '\n', text)
        text = re.sub(r'\n--- Tables ---\n', '\n', text)
        text = re.sub(r'\n--- EPUB Content ---\n', '\n', text)

        # Clean up any remaining HTML entities or tags that might have escaped
        text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)  # HTML entities
        text = re.sub(r'<[^>]+>', '', text)  # HTML tags

        return text.strip()

    def assess_text_quality(self, text: str) -> Dict[str, any]:
        """
        Assess the quality of extracted text (same as original)

        Args:
            text: Extracted text to assess

        Returns:
            Dictionary with quality metrics
        """
        if not text:
            return {
                'word_count': 0,
                'char_count': 0,
                'avg_word_length': 0,
                'line_count': 0,
                'quality_score': 0.0,
                'has_substantial_content': False
            }

        words = text.split()
        lines = text.split('\n')

        # Calculate basic metrics
        word_count = len(words)
        char_count = len(text)
        avg_word_length = sum(len(word) for word in words) / max(word_count, 1)
        line_count = len([line for line in lines if line.strip()])

        # Quality indicators
        quality_indicators = {
            'sufficient_length': word_count > 1000,
            'reasonable_word_length': 3 <= avg_word_length <= 15,
            'not_mostly_numbers': sum(1 for word in words[:100] if word.isdigit()) < 50,
            'has_sentences': '.' in text and ('?' in text or '!' in text),
            'not_repetitive': len(set(words[:100])) > 50 if len(words) >= 100 else True
        }

        quality_score = sum(quality_indicators.values()) / len(quality_indicators)

        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'line_count': line_count,
            'quality_score': quality_score,
            'has_substantial_content': quality_score > 0.6,
            'quality_indicators': quality_indicators
        }

    def get_file_type(self, file_path: str) -> str:
        """
        Determine file type based on extension

        Args:
            file_path: Path to file

        Returns:
            File type string ('pdf' or 'epub')
        """
        return Path(file_path).suffix.lower().replace('.', '')

    def extract_single_document(self, file_path: str, save_output: bool = True) -> ExtractionResult:
        """
        Extract text from a single document file (PDF or EPUB)

        Args:
            file_path: Path to document file
            save_output: Whether to save extracted text to file

        Returns:
            ExtractionResult object with extraction details
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        filename = file_path.name
        file_type = self.get_file_type(str(file_path))

        logger.info(f"Starting extraction for: {filename} (type: {file_type})")

        # Route to appropriate extraction methods based on file type
        if file_type == 'pdf':
            return self._extract_pdf(file_path, save_output, start_time)
        elif file_type == 'epub':
            return self._extract_epub(file_path, save_output, start_time)
        else:
            return ExtractionResult(
                filename=filename,
                success=False,
                text="",
                page_count=0,
                extraction_method="None",
                file_type=file_type,
                error_message=f"Unsupported file type: {file_type}",
                processing_time=(datetime.now() - start_time).total_seconds()
            )

    def _extract_pdf(self, pdf_path: Path, save_output: bool, start_time: datetime) -> ExtractionResult:
        """Extract text from PDF file using dual approach"""
        filename = pdf_path.name

        # Try PyPDF2 first
        text, pypdf2_success = self.extract_with_pypdf2(str(pdf_path))
        extraction_method = "PyPDF2"

        if not pypdf2_success:
            logger.info(f"PyPDF2 failed for {filename}, trying pdfplumber...")
            text, pdfplumber_success = self.extract_with_pdfplumber(str(pdf_path))
            extraction_method = "pdfplumber"

            if not pdfplumber_success:
                logger.error(f"Both PDF extraction methods failed for {filename}")
                self.extraction_stats['both_failed'] += 1
                return ExtractionResult(
                    filename=filename,
                    success=False,
                    text="",
                    page_count=0,
                    extraction_method="None",
                    file_type="pdf",
                    error_message="Both PyPDF2 and pdfplumber failed",
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

        return self._finalize_extraction(
            text, filename, extraction_method, "pdf",
            pdf_path, save_output, start_time
        )

    def _extract_epub(self, epub_path: Path, save_output: bool, start_time: datetime) -> ExtractionResult:
        """Extract text from EPUB file using dual approach"""
        filename = epub_path.name

        # Try ebooklib first
        text, ebooklib_success = self.extract_with_ebooklib(str(epub_path))
        extraction_method = "ebooklib"

        if not ebooklib_success:
            logger.info(f"ebooklib failed for {filename}, trying epub2txt...")
            text, epub2txt_success = self.extract_with_epub2txt(str(epub_path))
            extraction_method = "epub2txt"

            if not epub2txt_success:
                logger.error(f"Both EPUB extraction methods failed for {filename}")
                self.extraction_stats['both_failed'] += 1
                return ExtractionResult(
                    filename=filename,
                    success=False,
                    text="",
                    page_count=0,
                    extraction_method="None",
                    file_type="epub",
                    error_message="Both ebooklib and epub2txt failed",
                    processing_time=(datetime.now() - start_time).total_seconds()
                )

        return self._finalize_extraction(
            text, filename, extraction_method, "epub",
            epub_path, save_output, start_time
        )

    def _finalize_extraction(self, text: str, filename: str, extraction_method: str,
                             file_type: str, file_path: Path, save_output: bool,
                             start_time: datetime) -> ExtractionResult:
        """Finalize extraction process with cleaning, quality assessment, and saving"""

        # Clean the extracted text
        cleaned_text = self.clean_extracted_text(text)

        # Assess text quality
        quality_metrics = self.assess_text_quality(cleaned_text)

        # Count pages/chapters (approximate)
        if file_type == "pdf":
            page_count = text.count("--- Page") or 1
        else:  # epub
            page_count = text.count("--- Chapter") or 1

        # Create result object
        result = ExtractionResult(
            filename=filename,
            success=quality_metrics['has_substantial_content'],
            text=cleaned_text,
            page_count=page_count,
            extraction_method=extraction_method,
            file_type=file_type,
            word_count=quality_metrics['word_count'],
            processing_time=(datetime.now() - start_time).total_seconds()
        )

        # Update statistics
        if result.success:
            self.extraction_stats['successful_extractions'] += 1
            if extraction_method == "PyPDF2":
                self.extraction_stats['pypdf2_success'] += 1
            elif extraction_method == "pdfplumber":
                self.extraction_stats['pdfplumber_success'] += 1
            elif extraction_method == "ebooklib":
                self.extraction_stats['ebooklib_success'] += 1
            elif extraction_method == "epub2txt":
                self.extraction_stats['epub2txt_success'] += 1
        else:
            self.extraction_stats['failed_extractions'] += 1
            result.error_message = f"Low quality extraction (quality score: {quality_metrics['quality_score']:.2f})"

        # Update file type counters
        if file_type == "pdf":
            self.extraction_stats['pdf_files'] += 1
        elif file_type == "epub":
            self.extraction_stats['epub_files'] += 1

        # Save output if requested and successful
        if save_output and result.success:
            output_path = self.output_dir / f"{file_path.stem}.txt"
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Extracted from: {filename}\n")
                    f.write(f"# File type: {file_type.upper()}\n")
                    f.write(f"# Extraction method: {extraction_method}\n")
                    f.write(f"# Word count: {result.word_count}\n")
                    f.write(f"# Processing time: {result.processing_time:.2f}s\n\n")
                    f.write(cleaned_text)
                logger.info(f"Saved extracted text to: {output_path}")
            except Exception as e:
                logger.error(f"Failed to save extracted text: {e}")

        logger.info(f"Completed extraction for {filename} - Success: {result.success}")
        return result

    def batch_extract(self, document_directory: str, file_patterns: List[str] = None) -> List[ExtractionResult]:
        """
        Extract text from multiple document files in a directory

        Args:
            document_directory: Directory containing document files
            file_patterns: List of patterns to match files (default: ["*.pdf", "*.epub"])

        Returns:
            List of ExtractionResult objects
        """
        if file_patterns is None:
            file_patterns = ["*.pdf", "*.epub"]

        doc_dir = Path(document_directory)
        doc_files = []

        # Collect all matching files
        for pattern in file_patterns:
            doc_files.extend(list(doc_dir.glob(pattern)))

        if not doc_files:
            logger.warning(f"No document files found in {document_directory} matching patterns {file_patterns}")
            return []

        logger.info(f"Found {len(doc_files)} document files to process")

        results = []
        self.extraction_stats['total_files'] = len(doc_files)

        for i, doc_path in enumerate(doc_files, 1):
            logger.info(f"Processing file {i}/{len(doc_files)}: {doc_path.name}")

            try:
                result = self.extract_single_document(doc_path)
                results.append(result)

                # Log progress
                success_rate = (self.extraction_stats['successful_extractions'] / i) * 100
                logger.info(
                    f"Progress: {i}/{len(doc_files)} ({i / len(doc_files) * 100:.1f}%) - Success rate: {success_rate:.1f}%")

            except Exception as e:
                logger.error(f"Unexpected error processing {doc_path.name}: {e}")
                results.append(ExtractionResult(
                    filename=doc_path.name,
                    success=False,
                    text="",
                    page_count=0,
                    extraction_method="None",
                    file_type=self.get_file_type(str(doc_path)),
                    error_message=f"Unexpected error: {str(e)}"
                ))

        # Generate final report
        self.generate_extraction_report(results)
        return results

    def generate_extraction_report(self, results: List[ExtractionResult]) -> None:
        """
        Generate comprehensive extraction report (enhanced for both file types)

        Args:
            results: List of extraction results
        """
        report = {
            'extraction_summary': self.extraction_stats.copy(),
            'successful_files': [],
            'failed_files': [],
            'quality_metrics': {
                'total_words': 0,
                'avg_words_per_file': 0,
                'total_processing_time': 0
            },
            'file_type_breakdown': {
                'pdf': {'successful': 0, 'failed': 0, 'total_words': 0},
                'epub': {'successful': 0, 'failed': 0, 'total_words': 0}
            }
        }

        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        # Populate successful files
        for result in successful_results:
            report['successful_files'].append({
                'filename': result.filename,
                'file_type': result.file_type,
                'word_count': result.word_count,
                'page_count': result.page_count,
                'extraction_method': result.extraction_method,
                'processing_time': result.processing_time
            })
            report['quality_metrics']['total_words'] += result.word_count
            report['quality_metrics']['total_processing_time'] += result.processing_time

            # Update file type breakdown
            if result.file_type in report['file_type_breakdown']:
                report['file_type_breakdown'][result.file_type]['successful'] += 1
                report['file_type_breakdown'][result.file_type]['total_words'] += result.word_count

        # Populate failed files
        for result in failed_results:
            report['failed_files'].append({
                'filename': result.filename,
                'file_type': result.file_type,
                'error_message': result.error_message,
                'extraction_method': result.extraction_method
            })

            # Update file type breakdown
            if result.file_type in report['file_type_breakdown']:
                report['file_type_breakdown'][result.file_type]['failed'] += 1

        # Calculate averages
        if successful_results:
            report['quality_metrics']['avg_words_per_file'] = (
                    report['quality_metrics']['total_words'] / len(successful_results)
            )

        # Save report
        report_path = self.output_dir / "extraction_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("DOCUMENT EXTRACTION REPORT")
        print("=" * 60)
        print(f"Total files processed: {len(results)}")
        print(
            f"Successful extractions: {len(successful_results)} ({len(successful_results) / len(results) * 100:.1f}%)")
        print(f"Failed extractions: {len(failed_results)} ({len(failed_results) / len(results) * 100:.1f}%)")
        print(f"\nFile Type Breakdown:")
        print(f"  PDF files: {self.extraction_stats['pdf_files']}")
        print(f"  EPUB files: {self.extraction_stats['epub_files']}")
        print(f"\nExtraction Method Success:")
        print(f"  PyPDF2 successes: {self.extraction_stats['pypdf2_success']}")
        print(f"  pdfplumber successes: {self.extraction_stats['pdfplumber_success']}")
        print(f"  ebooklib successes: {self.extraction_stats['ebooklib_success']}")
        print(f"  epub2txt successes: {self.extraction_stats['epub2txt_success']}")
        print(f"\nQuality Metrics:")
        print(f"  Total words extracted: {report['quality_metrics']['total_words']:,}")
        print(f"  Average words per file: {report['quality_metrics']['avg_words_per_file']:.0f}")
        print(f"  Total processing time: {report['quality_metrics']['total_processing_time']:.2f}s")
        print(f"\nReport saved to: {report_path}")

        if failed_results:
            print(f"\nFailed files:")
            for result in failed_results:
                print(f"  - {result.filename} ({result.file_type}): {result.error_message}")


# Backward compatibility alias
PDFExtractor = DocumentExtractor


# Example usage and testing functions
def main():
    """Example usage of DocumentExtractor"""
    extractor = DocumentExtractor()

    # Test single file extraction
    # pdf_result = extractor.extract_single_document("data/raw_books/sample_book.pdf")
    # epub_result = extractor.extract_single_document("data/raw_books/sample_book.epub")
    # print(f"PDF result: {pdf_result}")
    # print(f"EPUB result: {epub_result}")

    # Test batch extraction for both PDF and EPUB files
    results = extractor.batch_extract("data/raw_books/")

    print(f"\nBatch extraction completed. Processed {len(results)} files.")


if __name__ == "__main__":
    main()