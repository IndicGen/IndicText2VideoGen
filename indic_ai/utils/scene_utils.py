import re
from typing import List

class ChunkHandler:
    def __init__(self, min_overlap=5):
        """
        Initialize the ChunkHandler.
        
        Args:
            min_overlap (int): Minimum characters to consider for detecting overlaps.
        """
        self.min_overlap = min_overlap
    
    def chunk_by_period(self,sentences: List[str]) -> List[str]:
        # Combine the list into a single text
        text = ' '.join(sentences)
        
        # Split based on periods only
        raw_chunks = re.split(r'\.', text)
        
        # Clean and filter empty chunks
        clean_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
        
        return clean_chunks

    def clean_text(self, text):
        """
        Clean up the merged text by fixing spaces and punctuation.
        """
        text = re.sub(r'\s+', ' ', text)               # Collapse multiple spaces/newlines
        text = re.sub(r'\s+([.,!?])', r'\1', text)      # Remove space before punctuation
        text = re.sub(r'([.,!?])([^\s])', r'\1 \2', text)  # Ensure space after punctuation
        return text.strip()

    def find_overlap(self, text1, text2):
        """
        Find the maximum overlap between end of text1 and start of text2.
        
        Returns:
            The overlapping string if found, else an empty string.
        """
        max_overlap_len = min(len(text1), len(text2))
        overlap = ""

        for i in range(self.min_overlap, max_overlap_len):
            if text1[-i:].lower() == text2[:i].lower():
                overlap = text1[-i:]
        
        return overlap

    def merge(self, chunks):
        """
        Merge a list of chunks intelligently, removing overlaps.
        
        Args:
            chunks (List[str]): List of text chunks.
        
        Returns:
            str: Coherent merged text.
        """
        if not chunks:
            return ""

        final_text = chunks[0].strip()

        for next_chunk in chunks[1:]:
            next_chunk = next_chunk.strip()
            overlap = self.find_overlap(final_text, next_chunk)

            if overlap:
                final_text += next_chunk[len(overlap):]
            else:
                if not final_text[-1] in ".!?":
                    final_text += "."
                final_text += " " + next_chunk

        return self.clean_text(final_text)
