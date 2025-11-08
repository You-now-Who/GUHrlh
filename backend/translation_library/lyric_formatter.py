import re
from typing import List, Dict

class LyricFormatter:
    @staticmethod
    def split_into_segments(text: str, max_segment_length: int = 1000) -> List[str]:
        """
        Split long lyrics into segments for better translation
        """
        # Split by verses/stanzas when possible
        segments = re.split(r'\n\s*\n', text)
        
        # Further split long segments
        result = []
        for segment in segments:
            if len(segment) <= max_segment_length:
                result.append(segment)
            else:
                # Split by lines but keep logical groups
                lines = segment.split('\n')
                current_segment = ""
                for line in lines:
                    if len(current_segment + line) > max_segment_length and current_segment:
                        result.append(current_segment.strip())
                        current_segment = line
                    else:
                        current_segment += line + "\n"
                if current_segment:
                    result.append(current_segment.strip())
        
        return result
    
    @staticmethod
    def reassemble_segments(translations: List[Dict]) -> str:
        """
        Reassemble translated segments back into full lyrics
        """
        return "\n\n".join([t['translated_text'] for t in translations])
    
    @staticmethod
    def preprocess_lyrics(text: str) -> str:
        """
        Clean and prepare lyrics for better translation
        """
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Strip whitespace from each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        return text.strip()