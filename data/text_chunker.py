#!/usr/bin/env python3
"""
Intelligent Text Chunker - Step 2.2
AI Brainstorming Agent - Foundation & Knowledge Base

Implements:
- Chapter/section detection
- Technique-aware chunking
- Context preservation with overlap
- Semantic boundary detection
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    chunk_id: str
    source_file: str
    start_position: int
    end_position: int
    token_count: int
    chunk_type: str  # 'chapter', 'section', 'technique', 'content'
    title: Optional[str] = None
    section_hierarchy: Optional[List[str]] = None
    overlap_with_previous: bool = False
    overlap_with_next: bool = False


class SemanticBoundaryDetector:
    """Detects semantic boundaries in text for intelligent chunking."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Chapter/section patterns
        self.chapter_patterns = [
            r'^(Chapter|CHAPTER)\s+(\d+|[IVX]+)[\s.:]\s*(.*)$',
            r'^(\d+)\.\s+(.+)$',
            r'^([IVX]+)\.\s+(.+)$',
            r'^# (.+)$',  # Markdown style
        ]

        self.section_patterns = [
            r'^(Section|SECTION)\s+(\d+[\.\d]*)\s*[-:]?\s*(.*)$',
            r'^(\d+\.\d+)\s+(.+)$',
            r'^## (.+)$',  # Markdown style
            r'^([A-Z][A-Z\s]{3,}?)$',  # ALL CAPS titles
        ]

        self.subsection_patterns = [
            r'^(\d+\.\d+\.\d+)\s+(.+)$',
            r'^### (.+)$',  # Markdown style
            r'^([a-z][\)\.])\s+(.+)$',  # a) or a.
        ]

        # Technique indicator patterns
        self.technique_patterns = [
            r'\b(technique|method|approach|strategy|process|procedure|algorithm)\b',
            r'\b(step\s+\d+|phase\s+\d+|stage\s+\d+)\b',
            r'\b(how\s+to|instructions?|guidelines?)\b',
            r'\b(example|illustration|case\s+study)\b',
            r'^\s*\d+[\.\)]\s+',  # Numbered lists
            r'^\s*[•\-\*]\s+',  # Bullet lists
        ]

        # Semantic break indicators
        self.break_indicators = [
            r'\n\s*\n\s*\n',  # Multiple line breaks
            r'\.{3,}',  # Ellipsis or dots
            r'^\s*\*\s*\*\s*\*\s*$',  # Asterisk separators
            r'^\s*-+\s*$',  # Dash separators
            r'^\s*=+\s*$',  # Equal sign separators
        ]

    def detect_structure_type(self, line: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Detect if a line is a chapter, section, or subsection.

        Returns:
            Tuple of (type, number, title)
        """
        line = line.strip()

        # Check for chapters
        for pattern in self.chapter_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 3:
                    return 'chapter', match.group(2), match.group(3)
                elif len(match.groups()) >= 2:
                    return 'chapter', match.group(1), match.group(2)
                else:
                    return 'chapter', None, match.group(1)

        # Check for sections
        for pattern in self.section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 3:
                    return 'section', match.group(2), match.group(3)
                elif len(match.groups()) >= 2:
                    return 'section', match.group(1), match.group(2)
                else:
                    return 'section', None, match.group(1)

        # Check for subsections
        for pattern in self.subsection_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) >= 2:
                    return 'subsection', match.group(1), match.group(2)
                else:
                    return 'subsection', None, match.group(1)

        return 'content', None, None

    def is_technique_content(self, text: str) -> bool:
        """Check if text contains technique-related content."""
        text_lower = text.lower()
        technique_score = 0

        for pattern in self.technique_patterns:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            technique_score += matches

        # If we find multiple technique indicators, it's likely technique content
        return technique_score >= 2

    def find_semantic_breaks(self, text: str) -> List[int]:
        """Find positions of semantic breaks in text."""
        breaks = []

        for pattern in self.break_indicators:
            for match in re.finditer(pattern, text, re.MULTILINE):
                breaks.append(match.start())

        return sorted(set(breaks))


class TokenEstimator:
    """Estimates token count for text chunks."""

    def __init__(self):
        # Rough estimation: 1 token ≈ 4 characters for English text
        self.chars_per_token = 4

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for given text."""
        # Remove extra whitespace for more accurate estimation
        clean_text = re.sub(r'\s+', ' ', text.strip())
        return max(1, len(clean_text) // self.chars_per_token)

    def find_token_boundary(self, text: str, target_tokens: int) -> int:
        """Find character position closest to target token count."""
        target_chars = target_tokens * self.chars_per_token

        if len(text) <= target_chars:
            return len(text)

        # Find sentence boundary near target
        search_start = max(0, target_chars - 200)
        search_end = min(len(text), target_chars + 200)
        search_text = text[search_start:search_end]

        # Look for sentence endings
        sentence_endings = []
        for match in re.finditer(r'[.!?]\s+', search_text):
            pos = search_start + match.end()
            sentence_endings.append(pos)

        if sentence_endings:
            # Return the sentence ending closest to target
            target_pos = target_chars
            closest = min(sentence_endings, key=lambda x: abs(x - target_pos))
            return closest

        # Fallback to word boundary
        word_boundary = text.rfind(' ', target_chars - 100, target_chars + 100)
        return word_boundary if word_boundary != -1 else target_chars


class IntelligentTextChunker:
    """Main chunking engine with semantic awareness."""

    def __init__(self,
                 min_chunk_size: int = 1000,
                 max_chunk_size: int = 1500,
                 overlap_size: int = 150,
                 preserve_techniques: bool = True):
        """
        Initialize the chunker.

        Args:
            min_chunk_size: Minimum chunk size in tokens
            max_chunk_size: Maximum chunk size in tokens
            overlap_size: Overlap size in tokens for context preservation
            preserve_techniques: Whether to preserve complete technique descriptions
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.preserve_techniques = preserve_techniques

        self.logger = logging.getLogger(__name__)
        self.boundary_detector = SemanticBoundaryDetector()
        self.token_estimator = TokenEstimator()

        self.chunks = []
        self.current_hierarchy = []

    def _create_overlap_text(self, previous_chunk: str, next_text: str) -> Tuple[str, str]:
        """Create overlap between chunks for context preservation."""
        overlap_tokens = self.overlap_size

        # Get overlap from end of previous chunk
        prev_overlap_chars = self.token_estimator.find_token_boundary(
            previous_chunk[::-1], overlap_tokens
        )
        prev_overlap = previous_chunk[-prev_overlap_chars:] if prev_overlap_chars > 0 else ""

        # Get overlap for beginning of next chunk
        next_overlap_chars = self.token_estimator.find_token_boundary(
            next_text, overlap_tokens
        )
        next_overlap = next_text[:next_overlap_chars] if next_overlap_chars > 0 else ""

        return prev_overlap, next_overlap

    def _detect_technique_boundaries(self, text: str) -> List[Tuple[int, int]]:
        """Detect complete technique descriptions to preserve."""
        if not self.preserve_techniques:
            return []

        boundaries = []
        lines = text.split('\n')
        current_technique_start = None

        for i, line in enumerate(lines):
            line_start = sum(len(lines[j]) + 1 for j in range(i))  # +1 for \n

            # Check if this line starts a technique
            if self.boundary_detector.is_technique_content(line):
                if current_technique_start is None:
                    current_technique_start = line_start
            else:
                # Check if we're ending a technique section
                if current_technique_start is not None:
                    # Look ahead to see if technique continues
                    next_lines = lines[i:i + 3]  # Check next 3 lines
                    continues = any(self.boundary_detector.is_technique_content(l)
                                    for l in next_lines)

                    if not continues:
                        line_end = line_start + len(line)
                        boundaries.append((current_technique_start, line_end))
                        current_technique_start = None

        # Handle technique that goes to end of text
        if current_technique_start is not None:
            boundaries.append((current_technique_start, len(text)))

        return boundaries

    def _create_chunk(self,
                      content: str,
                      chunk_id: str,
                      source_file: str,
                      start_pos: int,
                      chunk_type: str = 'content',
                      title: str = None) -> TextChunk:
        """Create a TextChunk object."""
        return TextChunk(
            content=content.strip(),
            chunk_id=chunk_id,
            source_file=source_file,
            start_position=start_pos,
            end_position=start_pos + len(content),
            token_count=self.token_estimator.estimate_tokens(content),
            chunk_type=chunk_type,
            title=title,
            section_hierarchy=self.current_hierarchy.copy()
        )

    def chunk_text(self, text: str, source_file: str) -> List[TextChunk]:
        """
        Chunk text into semantically meaningful pieces.

        Args:
            text: Input text to chunk
            source_file: Source filename for metadata

        Returns:
            List of TextChunk objects
        """
        self.logger.info(f"Chunking text from {source_file}")
        self.chunks = []
        self.current_hierarchy = []

        # Detect structural boundaries
        lines = text.split('\n')
        structure_points = []

        for i, line in enumerate(lines):
            line_start = sum(len(lines[j]) + 1 for j in range(i))
            struct_type, number, title = self.boundary_detector.detect_structure_type(line)

            if struct_type != 'content':
                structure_points.append({
                    'position': line_start,
                    'type': struct_type,
                    'number': number,
                    'title': title,
                    'line': line.strip()
                })

        # Detect technique boundaries
        technique_boundaries = self._detect_technique_boundaries(text)

        # Create chunks
        current_pos = 0
        chunk_counter = 1

        while current_pos < len(text):
            # Determine chunk end position
            remaining_text = text[current_pos:]

            # Check if we're near a structural boundary
            next_structure = None
            for sp in structure_points:
                if sp['position'] > current_pos:
                    next_structure = sp
                    break

            # Check for technique boundaries
            technique_end = None
            for start, end in technique_boundaries:
                if start >= current_pos and start < current_pos + self.max_chunk_size * 4:
                    technique_end = end
                    break

            # Determine chunk size
            if technique_end and technique_end - current_pos <= self.max_chunk_size * 4:
                # Preserve complete technique
                chunk_end = technique_end
                chunk_type = 'technique'
            elif next_structure and next_structure['position'] - current_pos <= self.max_chunk_size * 4:
                # End chunk before next major structure
                chunk_end = next_structure['position']
                chunk_type = 'content'
            else:
                # Use token-based boundary
                chunk_end = current_pos + self.token_estimator.find_token_boundary(
                    remaining_text, self.max_chunk_size
                )
                chunk_type = 'content'

            # Extract chunk content
            chunk_content = text[current_pos:chunk_end]

            # Skip if chunk is too small (unless it's the last chunk)
            token_count = self.token_estimator.estimate_tokens(chunk_content)
            if token_count < self.min_chunk_size and chunk_end < len(text):
                # Extend chunk to minimum size
                chunk_end = current_pos + self.token_estimator.find_token_boundary(
                    remaining_text, self.min_chunk_size
                )
                chunk_content = text[current_pos:chunk_end]

            # Update hierarchy based on structure points
            for sp in structure_points:
                if current_pos <= sp['position'] < chunk_end:
                    if sp['type'] == 'chapter':
                        self.current_hierarchy = [sp['title'] or sp['line']]
                    elif sp['type'] == 'section':
                        if len(self.current_hierarchy) == 0:
                            self.current_hierarchy = [sp['title'] or sp['line']]
                        else:
                            self.current_hierarchy = [self.current_hierarchy[0], sp['title'] or sp['line']]

            # Create chunk
            chunk_id = f"{Path(source_file).stem}_chunk_{chunk_counter:03d}"
            chunk = self._create_chunk(
                content=chunk_content,
                chunk_id=chunk_id,
                source_file=source_file,
                start_pos=current_pos,
                chunk_type=chunk_type
            )

            # Add overlap with previous chunk
            if self.chunks and self.overlap_size > 0:
                prev_overlap, next_overlap = self._create_overlap_text(
                    self.chunks[-1].content, chunk_content
                )

                if prev_overlap:
                    # Add overlap to current chunk
                    chunk.content = prev_overlap + "\n---\n" + chunk.content
                    chunk.overlap_with_previous = True

                # Mark previous chunk for next overlap
                if next_overlap:
                    self.chunks[-1].overlap_with_next = True

            self.chunks.append(chunk)

            # Move to next position
            current_pos = chunk_end
            chunk_counter += 1

            # Avoid infinite loops
            if chunk_end <= current_pos:
                current_pos += 1

        self.logger.info(f"Created {len(self.chunks)} chunks from {source_file}")
        return self.chunks

    def save_chunks_metadata(self, chunks: List[TextChunk], output_path: Path):
        """Save chunk metadata to JSON file."""
        metadata = []
        for chunk in chunks:
            metadata.append({
                'chunk_id': chunk.chunk_id,
                'source_file': chunk.source_file,
                'token_count': chunk.token_count,
                'chunk_type': chunk.chunk_type,
                'title': chunk.title,
                'section_hierarchy': chunk.section_hierarchy,
                'start_position': chunk.start_position,
                'end_position': chunk.end_position,
                'overlap_with_previous': chunk.overlap_with_previous,
                'overlap_with_next': chunk.overlap_with_next,
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Saved metadata for {len(chunks)} chunks to {output_path}")


# Main execution interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Intelligent Text Chunker - Step 2.2")
    parser.add_argument("input_dir", help="Directory with cleaned text files")
    parser.add_argument("-o", "--output_dir", help="Output directory for chunks")
    parser.add_argument("--min_size", type=int, default=1000, help="Minimum chunk size in tokens")
    parser.add_argument("--max_size", type=int, default=1500, help="Maximum chunk size in tokens")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap size in tokens")
    parser.add_argument("-l", "--log_level", choices=["DEBUG", "INFO", "WARNING"],
                        default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Initialize chunker
    chunker = IntelligentTextChunker(
        min_chunk_size=args.min_size,
        max_chunk_size=args.max_size,
        overlap_size=args.overlap,
        preserve_techniques=True
    )

    # Process files
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "chunks"
    output_dir.mkdir(parents=True, exist_ok=True)

    text_files = list(input_dir.glob("*.txt"))

    print(f"Found {len(text_files)} text files to chunk")

    all_chunks = []
    for file_path in text_files:
        # Read cleaned text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Create chunks
        chunks = chunker.chunk_text(text, file_path.name)

        # Save individual chunks
        for chunk in chunks:
            chunk_file = output_dir / f"{chunk.chunk_id}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(f"# {chunk.chunk_id}\n")
                f.write(f"Source: {chunk.source_file}\n")
                f.write(f"Type: {chunk.chunk_type}\n")
                f.write(f"Tokens: {chunk.token_count}\n")
                if chunk.section_hierarchy:
                    f.write(f"Section: {' > '.join(chunk.section_hierarchy)}\n")
                f.write("\n" + "=" * 50 + "\n\n")
                f.write(chunk.content)

        all_chunks.extend(chunks)

    # Save metadata
    chunker.save_chunks_metadata(all_chunks, output_dir / "chunks_metadata.json")

    print(f"\nChunking complete!")
    print(f"Created {len(all_chunks)} chunks")
    print(f"Average chunk size: {sum(c.token_count for c in all_chunks) / len(all_chunks):.0f} tokens")
    print(f"Chunks saved to: {output_dir}")