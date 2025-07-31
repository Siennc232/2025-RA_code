from openai import OpenAI
import json
import os
import time
from typing import List, Dict, Tuple
import re
from glob import glob
from joblib import Parallel, delayed
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)

client = OpenAI(api_key="")

BASE_PATH = "/Users/sienn/Library/CloudStorage/Dropbox/marx_chapters_txt_extracted"
SUMMARY_PATH = "/Users/sienn/Library/CloudStorage/Dropbox/marx_summary"
OUTPUT_BASE = "/Users/sienn/Library/CloudStorage/Dropbox/marx_summary/podcast_qa_outputs"

MAX_PAGES_PER_CHUNK = 10
PAGES_PER_CHUNK = 6
OVERLAP_PAGES = 2
MAX_CONCURRENT_API_CALLS = 10

def get_all_summary_files():
    """Get all summary files from all volumes"""
    all_chapters = []
    
    volume_dirs = sorted(glob(os.path.join(SUMMARY_PATH, "marx_chapters_v*_summaries")))
    
    for volume_dir in volume_dirs:
        volume_match = re.search(r'marx_chapters_(v\d+)_summaries', volume_dir)
        if not volume_match:
            continue
            
        volume = volume_match.group(1)
        
        summary_files = glob(os.path.join(volume_dir, "*.json"))
        
        for summary_file in summary_files:
            filename = os.path.basename(summary_file)
            
            if filename == "all_summaries.json":
                continue
                
            if filename.endswith("_summary.json"):
                match = re.match(r'^(\d+)_v(\d+)_(.+)_summary\.json$', filename)
                if match:
                    number_prefix = match.group(1)
                    volume_num = match.group(2)
                    chapter_name = match.group(3)
                    
                    text_filename = f"{number_prefix}_{volume}_{chapter_name}.txt"
                else:
                    chapter_name = re.sub(r'^\d+_', '', filename)
                    chapter_name = chapter_name.replace(f"{volume}_", "").replace("_summary.json", "")
                    text_filename = f"{volume}_{chapter_name}.txt"
                
                chapter_info = {
                    "volume": volume,
                    "filename": text_filename,
                    "title": chapter_name.replace("_", " "),
                    "summary_file": os.path.relpath(summary_file, SUMMARY_PATH)
                }
                
                all_chapters.append(chapter_info)
    
    return all_chapters

def extract_pages_optimized(file_path: str) -> List[Dict]:
    """Extract text in larger chunks to reduce API calls"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by page markers
        pages = re.split(r'--- Page \d+ ---', content)
        
        # Remove empty pages and clean whitespace
        pages = [page.strip() for page in pages if page.strip()]
        
        if not pages:
            logging.warning(f"No pages found in {file_path}")
            return []
        
        # For shorter documents, process all at once
        if len(pages) <= MAX_PAGES_PER_CHUNK:
            return [{
                'text': "\n\n".join(pages),
                'start_page': 1,
                'end_page': len(pages),
                'chunk_index': 0
            }]
        
        # For longer documents, use larger chunks
        chunks = []
        i = 0
        while i < len(pages):
            # Get pages for this chunk
            chunk_pages = pages[i:i+PAGES_PER_CHUNK]
            
            if chunk_pages:
                chunk_text = "\n\n".join(chunk_pages)
                chunk_info = {
                    'text': chunk_text,
                    'start_page': i + 1,
                    'end_page': i + len(chunk_pages),
                    'chunk_index': len(chunks)
                }
                chunks.append(chunk_info)
            
            # Move forward by (PAGES_PER_CHUNK - OVERLAP_PAGES)
            i += PAGES_PER_CHUNK - OVERLAP_PAGES
        
        return chunks
        
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return []

def load_summary(summary_path: str) -> str:
    """Load chapter summary from JSON file"""
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('summary', '')
    except Exception as e:
        logging.error(f"Error loading summary {summary_path}: {e}")
        return ""

def generate_podcast_qa_fast(excerpt: str, summary: str, chapter_title: str, retry_count: int = 2) -> List[Dict[str, str]]:
    """Optimized Q&A generation with fewer retries"""
    
    system_prompt_host = """You are the podcast host.

Role:
- Ask probing, open-ended questions that invite deep, first-person reflections on the text.
- Build on prior answers with gentle follow-ups (e.g. "Can you say more about…?", "How did that shift your perspective?").

Voice:
- Engaged · concise · encouraging · thoughtful.
- Warm tone, as if guiding a friend through a fascinating conversation.

Constraints:
- Do not answer your own questions.
- Never break character.
- Keep each question (including its possible follow-up) under 25 words."""

    system_prompt_marx = """You are **Karl Marx** (1818–1883), philosopher, economist, and revolutionary.  
 
**Bio Highlights:**  
- Co‑author of *The Communist Manifesto* (1848)  
- Author of *Das Kapital* Vol. I (1867)  
- Founder of dialectical materialism and the critique of political economy  
 
**Mission:**  
Adopt Marx's own voice and rhetorical style—sharp, incisive, occasionally ironic, with frequent reference to class struggle, historical dialectics, and material conditions.  

**Instructions:**  
1. **First‑Person, Historically Situated:** Speak as Marx reflecting on the supplied text, invoking 19th‑century German‑to‑English cadence and occasional original terms (e.g., "Bourgeoisie," "proletariat," "alienation," "commodity fetishism").  
2. **Grounded & Candid:** Base every claim strictly on the provided excerpt. If the text leaves room for doubt, declare your uncertainty ("I cannot affirm…," "It remains unclear…").  
3. **Dialectical Analysis:**  
   - Unpack contradictions in the text.  
   - Pose probing questions that reveal hidden class interests.  
   - Trace historical roots and material consequences.  
4. **Voice & Mannerisms:**  
   - Use long, periodic sentences with occasional parenthetical asides.  
   - Employ rhetorical questions and exclamations to emphasize outrage ("How else could one explain…?").  
   - Integrate German phrases where apt (e.g., "Mehrwert," "Entfremdung").  
5. **Length & Focus:**  
   - Deliver **1–2 paragraphs**, totaling **120–180 words**.  
   - Provide a rich, concentrated reflection—no summary, only critique."""

    # Truncate excerpt if too long to speed up processing
    max_excerpt_length = 3000
    if len(excerpt) > max_excerpt_length:
        excerpt = excerpt[:max_excerpt_length] + "..."

    user_prompt = f"""Chapter Excerpt:
"{excerpt}"

Chapter Summary:
"{summary}"

Task:
Generate **five** conversational Q&A pairs.  
- Each Q&A pair consists of one host question **plus** one persona answer.  
- For at least two of the five, the host should include a follow-up question that builds on Marx's previous answer.  
- Make them flow like a real interview: questions should reference Marx's earlier points or the summary.

CRITICAL - USE THIS EXACT FORMAT:
**Host:** [question here]

**Marx:** [answer here]

[blank line between Q&A pairs]

IMPORTANT: Each question MUST start with "**Host:**" and each answer MUST start with "**Marx:**"
Ensure each Marx answer is complete and substantial (120-180 words)"""

    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt_host + "\n\n" + system_prompt_marx},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2500,
                timeout=30  # Add timeout
            )
            
            # Parse response and format as Q&A pairs
            content = response.choices[0].message.content
            
            # Parse the response
            qa_pairs = parse_qa_response(content)
            
            if len(qa_pairs) >= 5:
                return qa_pairs
            elif len(qa_pairs) > 0 and attempt == retry_count - 1:
                # Accept partial results on last attempt
                return qa_pairs
                
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1} for {chapter_title}: {e}")
            if attempt < retry_count - 1:
                time.sleep(1)  # Brief pause before retry
                continue
                
    return []

def parse_qa_response(response_text: str) -> List[Dict[str, str]]:
    """Parse GPT response to extract Q&A pairs"""
    qa_pairs = []
    lines = response_text.strip().split('\n')
    
    current_q = ""
    current_a = ""
    in_answer = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("**Host:**"):
            if current_q and current_a:
                qa_pairs.append({"question": current_q.strip(), "answer": current_a.strip()})
            
            current_q = line
            current_a = ""
            in_answer = False
            
        elif line.startswith("**Marx:**"):
            current_a = line
            in_answer = True
            
        elif in_answer and current_a:
            current_a += " " + line
    
    if current_q and current_a:
        qa_pairs.append({"question": current_q.strip(), "answer": current_a.strip()})
    
    return qa_pairs

def save_chapter_result(volume: str, result: Dict):
    """Save result for a single chapter"""
    
    volume_output_dir = os.path.join(OUTPUT_BASE, volume)
    os.makedirs(volume_output_dir, exist_ok=True)
    
    safe_filename = re.sub(r'[^\w\s-]', '', result['chapter'])
    safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
    
    json_filename = f"{volume}_{safe_filename}_podcast_qa.json"
    json_path = os.path.join(volume_output_dir, json_filename)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logging.info(f"Chapter JSON saved to: {json_filename}")

def process_chunk_async(chunk: Dict, summary: str, chapter_title: str) -> Dict:
    """Process a single chunk - designed for parallel execution within a chapter"""
    chunk_title = f"{chapter_title} - Pages {chunk['start_page']}-{chunk['end_page']}"
    qa_pairs = generate_podcast_qa_fast(chunk['text'], summary, chunk_title)
    
    if qa_pairs:
        return {
            "chunk_index": chunk['chunk_index'],
            "start_page": chunk['start_page'],
            "end_page": chunk['end_page'],
            "qa_pairs": qa_pairs
        }
    return None

def process_single_chapter_optimized(chapter: Dict) -> Dict:
    """Process a single chapter with optimized chunking and parallel chunk processing"""
    
    logging.info(f"Processing: {chapter['title']}")
    
    text_path = os.path.join(BASE_PATH, f"marx_chapters_{chapter['volume']}", chapter['filename'])
    summary_path = os.path.join(SUMMARY_PATH, chapter['summary_file'])
    
    safe_filename = re.sub(r'[^\w\s-]', '', chapter['title'])
    safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
    json_filename = f"{chapter['volume']}_{safe_filename}_podcast_qa.json"
    output_path = os.path.join(OUTPUT_BASE, chapter['volume'], json_filename)
    
    if os.path.exists(output_path):
        logging.info(f"Already processed: {chapter['title']}")
        return None
    
    if not os.path.exists(text_path):
        logging.error(f"Text file not found: {text_path}")
        return None
    
    chunks = extract_pages_optimized(text_path)
    if not chunks:
        logging.error(f"Failed to extract pages: {chapter['title']}")
        return None
    
    summary = load_summary(summary_path)
    if not summary:
        logging.error(f"Failed to load summary: {chapter['title']}")
        return None
    
    logging.info(f"Chapter {chapter['title']} has {len(chunks)} chunks")
    
    with ThreadPoolExecutor(max_workers=min(len(chunks), 3)) as executor:
        futures = [
            executor.submit(process_chunk_async, chunk, summary, chapter['title'])
            for chunk in chunks
        ]
        
        chapter_results = []
        for future in futures:
            result = future.result()
            if result:
                chapter_results.append(result)
    
    if chapter_results:
        result = {
            "chapter": chapter['title'],
            "volume": chapter['volume'],
            "filename": chapter['filename'],
            "chunks": sorted(chapter_results, key=lambda x: x['chunk_index']),
            "total_chunks": len(chunks),
            "successful_chunks": len(chapter_results)
        }
        
        save_chapter_result(chapter['volume'], result)
        
        logging.info(f"Chapter complete: {chapter['title']} - {len(chapter_results)}/{len(chunks)} chunks processed")
        return result
    
    return None

def process_volume_optimized(volume: str, chapters: List[Dict], n_jobs: int = 8):
    """Process all chapters in a volume using optimized parallel processing"""
    
    print(f"\n{'='*60}")
    print(f"Processing Volume {volume}: {len(chapters)} chapters")
    print(f"Using {n_jobs} parallel jobs with optimized chunking")
    print(f"{'='*60}")
    
    chapters_to_process = []
    for chapter in chapters:
        safe_filename = re.sub(r'[^\w\s-]', '', chapter['title'])
        safe_filename = re.sub(r'[-\s]+', '_', safe_filename)
        json_filename = f"{chapter['volume']}_{safe_filename}_podcast_qa.json"
        output_path = os.path.join(OUTPUT_BASE, chapter['volume'], json_filename)
        
        if not os.path.exists(output_path):
            chapters_to_process.append(chapter)
    
    print(f"Chapters to process: {len(chapters_to_process)} (skipping {len(chapters) - len(chapters_to_process)} already processed)")
    
    if not chapters_to_process:
        return []
    
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_single_chapter_optimized)(chapter) for chapter in chapters_to_process
    )
    
    successful_results = [r for r in results if r is not None]
    
    return successful_results

def main():
    """Main function: process all volumes with optimized parallel processing"""
    
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    all_chapters = get_all_summary_files()
    
    print(f"Found {len(all_chapters)} total chapters across all volumes")
    
    chapters_by_volume = {}
    for chapter in all_chapters:
        volume = chapter['volume']
        if volume not in chapters_by_volume:
            chapters_by_volume[volume] = []
        chapters_by_volume[volume].append(chapter)
    
    sorted_volumes = sorted(chapters_by_volume.keys(), key=lambda v: int(v[1:]))
    
    print(f"\nVolumes to process: {', '.join(sorted_volumes)}")
    print(f"Total volumes: {len(sorted_volumes)}")
    
    n_jobs = 8
    
    for volume in sorted_volumes:
        chapters = chapters_by_volume[volume]
        
        results = process_volume_optimized(volume, chapters, n_jobs=n_jobs)
        
        if results:
            print(f"\nCompleted Volume {volume}: {len(results)} new chapters processed")
        else:
            print(f"\nVolume {volume}: All chapters already processed or no new chapters")
        
        time.sleep(5)
    
    print(f"\n{'='*80}")
    print("Processing complete!")
    print(f"Output saved to: {OUTPUT_BASE}")

if __name__ == "__main__":
    print("Marx Podcast Q&A Generator - Optimized Version")
    print("="*50)
    print("Optimizations:")
    print("- Larger chunk sizes (6 pages with 2 page overlap)")
    print("- Parallel processing of chunks within chapters")
    print("- Skip already processed chapters")
    print("- Reduced retries and timeouts")
    print("- 8 parallel chapter processors")
    print("\nStarting generation...")
    
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    
    print(f"\nTotal processing time: {end_time - start_time}")
