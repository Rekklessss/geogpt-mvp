"""
Text splitting and parsing utilities using BERT for intelligent document segmentation.

This module provides sophisticated text splitting that uses BERT's next sentence prediction
to make intelligent decisions about where to split documents while preserving semantic
coherence.
"""

from __future__ import annotations

import os
import re
import math
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional

import torch
from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer

try:
    from nltk import sent_tokenize
except ImportError:
    import nltk
    nltk.download('punkt')
    from nltk import sent_tokenize

from ..config import BERT_PATH, MAX_SIZE


# Document sections to recognize
SECTIONS = ['abstract', 'introduction', 'conclusion', 'acknowledgement', 'reference']
FILTERED_SECTIONS = ['acknowledgement', 'reference', 'acknowledgment']


class TextSplitter:
    """
    Intelligent text splitter using BERT for next sentence prediction.
    
    Provides document structure-aware splitting that maintains semantic coherence
    while respecting token limits.
    """
    
    def __init__(
        self,
        model_name: str = BERT_PATH,
        max_size: int = MAX_SIZE,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the text splitter.
        
        Args:
            model_name: BERT model path/name for tokenization and prediction
            max_size: Maximum token size per chunk
            device: Device to run model on (auto-detected if None)
        """
        self.max_size = max_size
        
        # Device detection following project patterns
        if device is None:
            device = os.getenv("TEXT_SPLITTER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        # Initialize BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForNextSentencePrediction.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.eval()

    def _model_predict(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]], 
        token_type_ids: List[List[int]],
        batch_size: int = 64
    ) -> List[List[Any]]:
        """Predict with automatic batch size reduction on OOM."""
        try:
            return self._model_result(input_ids, attention_mask, token_type_ids, batch_size)
        except RuntimeError:
            torch.cuda.empty_cache()
            try:
                return self._model_result(input_ids, attention_mask, token_type_ids, batch_size // 2)
            except RuntimeError:
                return self._model_result(input_ids, attention_mask, token_type_ids, batch_size // 4)

    def _model_result(
        self,
        input_ids: List[List[int]],
        attention_mask: List[List[int]],
        token_type_ids: List[List[int]],
        batch_size: int
    ) -> List[List[Any]]:
        """Run BERT model inference in batches."""
        self.model.eval()
        with torch.no_grad():
            all_probs = []
            batch_num = math.ceil(len(input_ids) / batch_size)
            
            for i in range(batch_num):
                start = batch_size * i
                end = min(start + batch_size, len(input_ids))

                t_input_ids = torch.tensor(input_ids[start:end]).to(self.device)
                t_attention_mask = torch.tensor(attention_mask[start:end]).to(self.device)
                t_token_type_ids = torch.tensor(token_type_ids[start:end]).to(self.device)

                outputs = self.model(
                    input_ids=t_input_ids,
                    attention_mask=t_attention_mask,
                    token_type_ids=t_token_type_ids
                )
                logits = outputs.logits
                probs = softmax(logits, dim=1).tolist()
                all_probs.extend(probs)

        return [[prob.index(max(prob)), prob[0]] for prob in all_probs]

    def _reset_chunk(
        self,
        new_chunks: List[List[Tuple]],
        sentences: List[Tuple],
        max_token_size: int
    ) -> None:
        """Recursively split chunks that exceed token limit."""
        if len(sentences) < 1:
            return
        if len(sentences) == 1:
            new_chunks.append(sentences)
            return

        total_tokens = sum(t_size for _, _, t_size, _ in sentences)
        if total_tokens <= max_token_size:
            new_chunks.append(sentences)
            return

        # Find best split point based on scores
        scores = [score for _, _, _, score in sentences[1:]]
        min_score = min(scores)
        split_idx = scores.index(min_score) + 1

        left_sentences = sentences[:split_idx]
        right_sentences = sentences[split_idx:]

        self._reset_chunk(new_chunks, left_sentences, max_token_size)
        self._reset_chunk(new_chunks, right_sentences, max_token_size)

    def _sentence_counter(self, text: str, counter: Dict[str, int]) -> Dict[str, int]:
        """Count bracket occurrences for sentence completion detection."""
        for c in text:
            if c in counter:
                counter[c] += 1
        return counter

    def _concat_sentences(self, sentences: List[str]) -> List[str]:
        """Concatenate sentences while respecting bracket pairing."""
        n_sentences = []
        temp = []
        counter = {"(": 0, "[": 0, ")": 0, "]": 0}
        
        for sent in sentences:
            counter = self._sentence_counter(sent, counter)
            temp.append(sent)
            
            # Continue if brackets are unmatched and not too many sentences
            if (counter["("] > counter[")"] or counter["["] > counter["]"]) and len(temp) < 10:
                continue
            else:
                n_sentences.append(temp)
                temp = []
                counter = {"(": 0, "[": 0, ")": 0, "]": 0}
        
        return [" ".join(s) for s in n_sentences]

    def _merge_chunks(self, chunks: List[List[Tuple]], max_token_size: int) -> List[List[Tuple]]:
        """Merge small chunks together up to token limit."""
        prev_sum_size = 0
        n_chunks = []
        cur_chunk = []
        
        for chunk in chunks:
            total_size = sum(int(t) for _, _, t, _ in chunk)
            
            if prev_sum_size + total_size > max_token_size:
                if cur_chunk:
                    n_chunks.append(cur_chunk)
                cur_chunk = chunk
                prev_sum_size = total_size
            else:
                cur_chunk.extend(chunk)
                prev_sum_size += total_size
        
        if cur_chunk:
            n_chunks.append(cur_chunk)
        
        return n_chunks

    def _split_long_paragraph(self, text: str) -> List[Tuple[str, int]]:
        """Split long paragraphs using BERT next sentence prediction."""
        sentences = sent_tokenize(text)
        sentences = self._concat_sentences(sentences)
        sentences = [s.strip() for s in sentences if s]
        
        if len(sentences) <= 1:
            return [(s, len(self.tokenizer(s).input_ids)) for s in sentences]

        # Initialize results and get token sizes
        results = [[1, 1]] * len(sentences)
        tok_sizes = [len(self.tokenizer(s).input_ids) for s in sentences]
        
        # Prepare sentence pairs for BERT
        former_sen = sentences[:len(sentences) - 1]
        latter_sen = sentences[1:]

        tokens = self.tokenizer(
            former_sen, 
            latter_sen, 
            padding='max_length', 
            max_length=512, 
            return_tensors=None
        )
        
        # Filter out sequences that are too long
        input_ids = []
        attention_mask = []
        token_type_ids = []
        sent_ids = []
        
        for i, input_id in enumerate(tokens.input_ids):
            if len(input_id) <= 512:
                sent_ids.append(i + 1)
                input_ids.append(input_id)
                attention_mask.append(tokens.attention_mask[i])
                token_type_ids.append(tokens.token_type_ids[i])

        # Get model predictions
        model_results = self._model_predict(input_ids, attention_mask, token_type_ids)

        # Update results with model predictions
        for i, mr in zip(sent_ids, model_results):
            results[i] = mr

        # Group sentences into chunks based on predictions
        predict_chunks = []
        for i, (sen, t_size, (relation, score)) in enumerate(zip(sentences, tok_sizes, results)):
            if not predict_chunks:
                predict_chunks.append([[i, sen, t_size, score]])
            else:
                predict_chunks[-1].append([i, sen, t_size, score])

        # Reset chunks that are too large
        reset_chunks = []
        for sentence_info in predict_chunks:
            temp_chunks = []
            self._reset_chunk(temp_chunks, sentence_info, self.max_size)
            reset_chunks.extend(temp_chunks)

        # Merge small chunks and sort
        reset_chunks = self._merge_chunks(reset_chunks, self.max_size)
        reset_chunks = [sorted(tc, key=lambda x: x[0]) for tc in reset_chunks]
        reset_chunks = sorted(reset_chunks, key=lambda x: x[0][0])
        
        # Create final output chunks
        out_chunks = [
            (" ".join([s for _, s, _, _ in chunk]), sum([int(t) for _, _, t, _ in chunk]))
            for chunk in reset_chunks
        ]
        
        return out_chunks

    def _calculate_section_level(self, section: str) -> int:
        """Calculate markdown header level."""
        num = 0
        for c in section.strip():
            if c == '#':
                num += 1
            else:
                break
        return num

    def _check_section(self, text: str) -> bool:
        """Check if text contains known section keywords."""
        return any(section in text.lower() for section in SECTIONS)

    def _reset_section_dict(self, sections_dict: Dict[int, Tuple[str, str]]) -> Dict[int, Tuple[str, str]]:
        """Reset sections if no main sections were identified."""
        # If no main sections identified, move subsections to main
        if all(sections[0] == '' for sections in sections_dict.values()):
            return {i: (sections[1], '') for i, sections in sections_dict.items()}
        return sections_dict

    def _sentence_complete(self, text: str) -> bool:
        """Check if sentence has balanced parentheses."""
        left_count = text.count('(')
        right_count = text.count(')')
        return right_count >= left_count

    def _filter_section(self, text: str) -> bool:
        """Filter out unwanted sections like references."""
        return any(filtered in text.strip().lower() for filtered in FILTERED_SECTIONS)

    def _concat_text(self, prev: str, new: str) -> str:
        """Concatenate text with proper spacing."""
        return prev + new if prev.endswith('-') else prev + ' ' + new

    def _concat_element(self, element: str, n_list: List[str]) -> List[str]:
        """Concatenate element with previous if conditions are met."""
        # If previous sentence appears incomplete, concatenate directly
        if n_list and (not n_list[-1].endswith('.') or not self._sentence_complete(n_list[-1])):
            n_list[-1] = self._concat_text(n_list[-1], element)
            return n_list
        
        # If both segments are short, concatenate with newline
        if n_list and (len(n_list[-1]) < 100 or len(element) < 100):
            n_list[-1] = n_list[-1] + "\n" + element
            return n_list
        
        n_list.append(element)
        return n_list

    def _merge_elements(self, elements: List[str]) -> List[str]:
        """Merge text elements while handling tables and concatenation rules."""
        n_list = []
        table_active = False
        
        for element in elements:
            if not element:
                continue
            
            # Handle tables - concatenate table rows
            if re.search(r'\|(.+)\|', element, re.M | re.I):
                if not table_active:
                    n_list.append(element)
                    table_active = True
                else:
                    n_list[-1] = n_list[-1] + '\n' + element
                continue
            elif table_active:
                table_active = False
                n_list.append(element)
                continue

            n_list = self._concat_element(element, n_list)
        
        return n_list

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs and remove duplicates."""
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_paragraphs = []
        for p in paragraphs:
            if p not in seen:
                unique_paragraphs.append(p)
                seen.add(p)
        
        return unique_paragraphs

    def _extract_title(self, info: Dict[str, Any], element: str) -> Tuple[Dict[str, Any], bool]:
        """Extract document title from markdown headers."""
        if self._calculate_section_level(element) == 1:
            info["title"] = element.lstrip('#').strip()
            info["s_index"] = 0
            info["section"] = ""
            info["subsection"] = ""
            info["sections"] = {0: ('', '')}
            info["raw_dict"] = defaultdict(list)
            return info, True
        return info, False

    def _extract_section(self, info: Dict[str, Any], match: Any, element: str) -> Tuple[Dict[str, Any], bool]:
        """Extract main section names."""
        if match and (self._check_section(element.lstrip('#').strip().lower()) or 
                     self._calculate_section_level(element) == 2):
            info["section"] = element.lstrip('#').strip()
            info["subsection"] = ''
            info["s_index"] += 1
            info["sections"][info["s_index"]] = (info["section"], info["subsection"])
            return info, True
        return info, False

    def _extract_subsection(
        self, 
        info: Dict[str, Any], 
        bold_match: Any, 
        is_bold: bool, 
        element: str
    ) -> Tuple[Dict[str, Any], bool]:
        """Extract subsection names."""
        if self._calculate_section_level(element) == 3 or is_bold:
            if is_bold:
                start, end = bold_match.span()
                info["subsection"] = element[start:end]
                element = element[end:].strip()
            else:
                info["subsection"] = element.lstrip('#').strip()
                element = ""
            
            info["s_index"] += 1
            info["sections"][info["s_index"]] = (info["section"], info["subsection"])
            
            if element:
                info["raw_dict"][info["s_index"]].append(element)
            
            return info, True
        return info, False

    def _extract_document_structure(self, paragraphs: List[str]) -> Dict[str, Any]:
        """Extract hierarchical structure from document."""
        info = {
            "raw_dict": defaultdict(list),
            "sections": {0: ('', '')},
            "title": "",
            "s_index": 0,
            "section": "",
            "subsection": ""
        }

        for element in paragraphs:
            bold_match = re.search(r'\*{2}.{2,}\*{2}', element, re.M | re.I)
            header_match = re.search(r'#+ ', element, re.M | re.I)
            
            is_bold_header = (bold_match and 
                            bold_match.span()[0] == 0 and 
                            bold_match.span()[1] < 50)

            if header_match or is_bold_header:
                # Try to extract title
                info, processed = self._extract_title(info, element)
                if processed:
                    continue

                # Try to extract main section
                info, processed = self._extract_section(info, header_match, element)
                if processed:
                    continue

                # Try to extract subsection
                info, processed = self._extract_subsection(info, bold_match, is_bold_header, element)
                if processed:
                    continue

            info["raw_dict"][info["s_index"]].append(element)

        return info

    def _refine_document_structure(self, info: Dict[str, Any]) -> Tuple[Dict[int, Tuple[str, str]], Dict[int, List[str]], List[str]]:
        """Refine document structure by removing empty sections and merging paragraphs."""
        sections = self._reset_section_dict(info["sections"])
        section_dict = {}
        all_sections = []
        
        for idx in sections:
            if idx in info["raw_dict"]:
                # Stop processing if we hit filtered sections
                if self._filter_section(sections[idx][0]):
                    break
                
                section_list = self._merge_elements(info["raw_dict"][idx])
                section_dict[idx] = section_list
                all_sections.extend(section_list)
        
        return sections, section_dict, all_sections

    def split_text(self, text: str, filename: str = "") -> List[Dict[str, Any]]:
        """
        Split text into structured chunks with metadata.
        
        Args:
            text: Input text to split
            filename: Source filename for metadata
            
        Returns:
            List of dictionaries containing chunk data with metadata
        """
        paragraphs = self._split_paragraphs(text)
        structure_info = self._extract_document_structure(paragraphs)
        sections, section_dict, all_sections = self._refine_document_structure(structure_info)

        data = []
        chunk_index = 0
        
        for section_idx, section_list in section_dict.items():
            combined_paragraph = "\n".join(section_list)
            total_tokens = len(self.tokenizer(combined_paragraph).input_ids)
            
            # Split long paragraphs using BERT
            if total_tokens > self.max_size:
                new_chunks = self._split_long_paragraph(combined_paragraph)
            else:
                new_chunks = [(combined_paragraph, total_tokens)]
            
            # Create structured output
            for chunk_text, token_count in new_chunks:
                data.append({
                    'title': structure_info["title"],
                    'section': sections[section_idx][0],
                    'subsection': sections[section_idx][1],
                    'source': filename,
                    'index': chunk_index,
                    'text': chunk_text,
                    'length': token_count
                })
                chunk_index += 1

        return data


# Convenience function for backward compatibility
def split_text(text: str, filename: str = "", max_size: int = MAX_SIZE) -> List[Dict[str, Any]]:
    """
    Convenience function to split text using default TextSplitter.
    
    Args:
        text: Input text to split
        filename: Source filename for metadata  
        max_size: Maximum token size per chunk
        
    Returns:
        List of dictionaries containing chunk data with metadata
    """
    splitter = TextSplitter(max_size=max_size)
    return splitter.split_text(text, filename)
