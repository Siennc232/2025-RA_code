#!/usr/bin/env python3
"""
Final Marx Text Files Summarizer
Generates accurate descriptive summaries (50 words) for Marx text files with improved content type detection.
Creates individual summary files organized by volume folders.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import openai
from openai import OpenAI
import time

class FinalMarxTextSummarizer:
    def __init__(self, api_key: str = None):
        """Initialize the summarizer with OpenAI API key."""
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Try to get from environment variable
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                self.client = None
                print("Warning: No OpenAI API key provided. Will use final basic text analysis only.")
    
    def extract_content_from_file(self, file_path: str) -> Dict[str, str]:
        """Extract and parse content from a Marx text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse the file structure
            lines = content.split('\n')
            
            # Extract metadata
            title = ""
            source = ""
            folder = ""
            content_text = ""
            
            in_content = False
            for line in lines:
                if line.startswith('Title:'):
                    title = line.replace('Title:', '').strip()
                elif line.startswith('Source:'):
                    source = line.replace('Source:', '').strip()
                elif line.startswith('Folder:'):
                    folder = line.replace('Folder:', '').strip()
                elif line.startswith('Content:'):
                    in_content = True
                    continue
                elif in_content and line.strip():
                    if line.startswith('--- Page'):
                        continue
                    content_text += line + '\n'
            
            return {
                'title': title,
                'source': source,
                'folder': folder,
                'content': content_text.strip(),
                'file_path': file_path
            }
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def analyze_content_structure(self, content: str, title: str) -> Dict[str, any]:
        """Analyze the structure and content type of the text with final improved detection."""
        analysis = {
            'content_type': 'unknown',
            'main_topics': [],
            'key_concepts': [],
            'writing_style': 'unknown',
            'length': len(content),
            'has_poetry': False,
            'has_philosophy': False,
            'has_economics': False,
            'has_politics': False,
            'has_letters': False,
            'has_correspondence': False,
            'has_capital': False,
            'has_correspondence_letters': False
        }
        
        content_lower = content.lower()
        title_lower = title.lower()
        
        # Priority-based content type detection
        # 1. Check for Capital/economic content first (highest priority)
        if any(word in title_lower for word in ['commodity', 'capital', 'value', 'political economy', 'production']):
            analysis['has_economics'] = True
            analysis['has_capital'] = True
            analysis['content_type'] = 'economics'
        elif any(word in content_lower for word in ['commodities and money', 'use value', 'exchange value', 'labor time', 'surplus value', 'capitalist mode of production']):
            analysis['has_economics'] = True
            analysis['has_capital'] = True
            analysis['content_type'] = 'economics'
        
        # 2. Check for correspondence/letters
        elif any(word in title_lower for word in ['letter', 'correspondence', 'dear']):
            analysis['has_letters'] = True
            analysis['has_correspondence'] = True
            analysis['has_correspondence_letters'] = True
            analysis['content_type'] = 'correspondence'
        elif any(word in content_lower for word in ['dear father', 'dear friend', 'yours sincerely', 'berlin, november', 'dear jenny']):
            analysis['has_letters'] = True
            analysis['has_correspondence'] = True
            analysis['content_type'] = 'correspondence'
        
        # 3. Check for poetry
        elif any(word in title_lower for word in ['verse', 'poem', 'ballad', 'sonnet', 'song', 'poetry']):
            analysis['has_poetry'] = True
            analysis['content_type'] = 'poetry'
        elif any(word in content_lower for word in ['verse', 'poem', 'ballad', 'sonnet', 'stanza', 'rhyme', 'dedicated to my dear father']):
            analysis['has_poetry'] = True
            analysis['content_type'] = 'poetry'
        
        # 4. Check for philosophy
        elif any(word in title_lower for word in ['hegel', 'philosophy', 'critique', 'dialectic', 'feuerbach']):
            analysis['has_philosophy'] = True
            analysis['content_type'] = 'philosophy'
        elif any(word in content_lower for word in ['hegel', 'dialectic', 'idealism', 'materialism', 'critique of religion', 'opium of the people']):
            analysis['has_philosophy'] = True
            analysis['content_type'] = 'philosophy'
        
        # 5. Check for politics
        elif any(word in title_lower for word in ['revolution', 'class', 'bourgeoisie', 'proletariat', 'state', 'communist']):
            analysis['has_politics'] = True
            analysis['content_type'] = 'politics'
        elif any(word in content_lower for word in ['bourgeoisie', 'proletariat', 'class struggle', 'revolution', 'state']):
            analysis['has_politics'] = True
            analysis['content_type'] = 'politics'
        
        # Extract key concepts with improved detection
        key_terms = [
            'marx', 'engels', 'hegel', 'capital', 'commodity', 'value', 'labor', 
            'class', 'bourgeoisie', 'proletariat', 'revolution', 'dialectic',
            'materialism', 'idealism', 'philosophy', 'economics', 'politics',
            'use value', 'exchange value', 'surplus value', 'alienation',
            'communism', 'socialism', 'feuerbach', 'religion', 'state'
        ]
        
        found_terms = [term for term in key_terms if term in content_lower]
        analysis['key_concepts'] = found_terms[:5]  # Top 5
        
        return analysis
    
    def generate_summary_with_openai(self, file_info: Dict[str, str], analysis: Dict[str, any]) -> str:
        """Generate summary using OpenAI API."""
        if not self.client:
            return self.generate_final_basic_summary(file_info, analysis)
        
        try:
            # Create a comprehensive prompt
            prompt = f"""
You are an expert scholar of Karl Marx's works. Please analyze the following text and provide a concise, accurate summary of approximately 50 words.

TEXT INFORMATION:
Title: {file_info['title']}
Content Type: {analysis['content_type']}
Key Concepts: {', '.join(analysis['key_concepts'])}
Length: {analysis['length']} characters

CONTENT:
{file_info['content'][:2000]}  # Limit to first 2000 chars for API efficiency

INSTRUCTIONS:
1. Focus on the main argument, theme, or purpose of the text
2. Identify the key ideas and concepts discussed
3. Note the historical context if relevant
4. Mention the form (poetry, philosophy, economics, correspondence, etc.)
5. Keep the summary to approximately 50 words
6. Be accurate and scholarly in tone
7. If the content seems to contain multiple distinct sections, focus on the primary theme

Please provide only the summary, no additional commentary.
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert scholar of Karl Marx's works, specializing in accurate textual analysis and summarization."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
            
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return self.generate_final_basic_summary(file_info, analysis)
    
    def generate_final_basic_summary(self, file_info: Dict[str, str], analysis: Dict[str, any]) -> str:
        """Generate a final improved basic summary without OpenAI API."""
        title = file_info['title']
        content = file_info['content']
        content_type = analysis['content_type']
        
        # Clean up title for better presentation
        clean_title = title.replace('v1_', '').replace('v3_', '').replace('v35_', '').replace('K_Marx_', '').replace('_', ' ')
        
        # Extract meaningful content based on type
        if content_type == 'poetry':
            # Look for dedication or first meaningful lines
            if 'dedicated to my dear father' in content.lower():
                summary = f"{clean_title}: Poetry collection by Marx dedicated to his father in 1837, containing verses and ballads."
            else:
                lines = content.split('\n')
                meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]
                if meaningful_lines:
                    first_line = meaningful_lines[0][:60] + "..." if len(meaningful_lines[0]) > 60 else meaningful_lines[0]
                    summary = f"{clean_title}: Poetry collection by Marx. {first_line}"
                else:
                    summary = f"{clean_title}: Poetry collection by Marx from his early years."
        
        elif content_type == 'correspondence':
            # Extract letter content
            if 'dear father' in content.lower():
                summary = f"{clean_title}: Personal letter from Marx to his father (1837) discussing his studies, poetry, and philosophical development."
            elif 'dear' in content.lower():
                summary = f"{clean_title}: Correspondence by Marx discussing {', '.join(analysis['key_concepts'][:3])}."
            else:
                summary = f"{clean_title}: Correspondence by Marx on {', '.join(analysis['key_concepts'][:3])}."
        
        elif content_type == 'economics':
            if 'commodity' in content.lower() and 'use value' in content.lower():
                summary = f"{clean_title}: Marx's analysis of commodities, use value, exchange value, and the labor theory of value from Capital."
            elif 'capital' in content.lower():
                summary = f"{clean_title}: Economic analysis by Marx focusing on {', '.join(analysis['key_concepts'][:3])}."
            else:
                summary = f"{clean_title}: Economic work by Marx examining {', '.join(analysis['key_concepts'][:3])}."
        
        elif content_type == 'philosophy':
            if 'hegel' in content.lower():
                summary = f"{clean_title}: Marx's critique of Hegel's philosophy, particularly his dialectical method and idealist approach."
            elif 'religion' in content.lower():
                summary = f"{clean_title}: Marx's critique of religion as the 'opium of the people' and its role in society."
            else:
                summary = f"{clean_title}: Philosophical work by Marx addressing {', '.join(analysis['key_concepts'][:3])}."
        
        elif content_type == 'politics':
            summary = f"{clean_title}: Political analysis by Marx examining {', '.join(analysis['key_concepts'][:3])} and class relations."
        
        else:
            # Generic summary
            summary = f"{clean_title}: {content_type.capitalize()} work by Marx focusing on {', '.join(analysis['key_concepts'][:3])}."
        
        return summary[:200]  # Limit length
    
    def process_file(self, file_path: str) -> Dict[str, any]:
        """Process a single file and generate summary."""
        print(f"Processing: {file_path}")
        
        # Extract content
        file_info = self.extract_content_from_file(file_path)
        if not file_info:
            return None
        
        # Analyze content
        analysis = self.analyze_content_structure(file_info['content'], file_info['title'])
        
        # Generate summary
        summary = self.generate_summary_with_openai(file_info, analysis)
        
        return {
            'file_path': file_path,
            'title': file_info['title'],
            'content_type': analysis['content_type'],
            'key_concepts': analysis['key_concepts'],
            'summary': summary,
            'word_count': len(summary.split()),
            'analysis': analysis
        }
    
    def create_summary_file(self, result: Dict[str, any], output_dir: str):
        """Create individual summary file for each processed file."""
        # Get the original file path and create corresponding summary file path
        original_path = Path(result['file_path'])
        
        # Extract volume folder (e.g., marx_chapters_v1)
        volume_folder = original_path.parent.name
        
        # Create output directory structure
        volume_output_dir = os.path.join(output_dir, volume_folder)
        os.makedirs(volume_output_dir, exist_ok=True)
        
        # Create summary filename (replace .txt with _summary.txt)
        summary_filename = original_path.stem + "_summary.txt"
        summary_file_path = os.path.join(volume_output_dir, summary_filename)
        
        # Create summary content
        summary_content = f"""MARX TEXT SUMMARY
{'='*50}

ORIGINAL FILE: {result['file_path']}
TITLE: {result['title']}
CONTENT TYPE: {result['content_type']}
KEY CONCEPTS: {', '.join(result['key_concepts'])}
WORD COUNT: {result['word_count']}

SUMMARY:
{result['summary']}

ANALYSIS:
- Content Type: {result['content_type']}
- Length: {result['analysis']['length']} characters
- Has Poetry: {result['analysis']['has_poetry']}
- Has Philosophy: {result['analysis']['has_philosophy']}
- Has Economics: {result['analysis']['has_economics']}
- Has Politics: {result['analysis']['has_politics']}
- Has Correspondence: {result['analysis']['has_correspondence']}

GENERATED BY: Marx Text Summarizer
DATE: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Write summary file
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"Created summary file: {summary_file_path}")
        return summary_file_path
    
    def process_directory(self, directory_path: str, output_dir: str = None) -> List[Dict[str, any]]:
        """Process all text files in a directory and its subdirectories, creating individual summary files."""
        if output_dir is None:
            # Default to Downloads directory
            downloads_path = os.path.expanduser("~/Downloads")
            output_dir = os.path.join(downloads_path, "marx_summaries")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        summary_files_created = []
        
        # Find all .txt files
        txt_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.txt') and not file.endswith('_summary.txt'):
                    txt_files.append(os.path.join(root, file))
        
        print(f"Found {len(txt_files)} text files to process")
        print(f"Output directory: {output_dir}")
        
        # Process files
        for i, file_path in enumerate(txt_files, 1):
            print(f"\nProcessing file {i}/{len(txt_files)}: {os.path.basename(file_path)}")
            
            result = self.process_file(file_path)
            if result:
                results.append(result)
                
                # Create individual summary file
                summary_file_path = self.create_summary_file(result, output_dir)
                summary_files_created.append(summary_file_path)
            
            # Add delay to avoid rate limiting
            if self.client:
                time.sleep(1)
        
        # Save comprehensive results
        comprehensive_results_file = os.path.join(output_dir, "comprehensive_summaries.json")
        with open(comprehensive_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Create summary report
        self.create_summary_report(results, output_dir)
        
        print(f"\n" + "="*80)
        print(f"PROCESSING COMPLETE")
        print(f"="*80)
        print(f"Processed {len(results)} files")
        print(f"Created {len(summary_files_created)} individual summary files")
        print(f"Output directory: {output_dir}")
        print(f"Comprehensive results: {comprehensive_results_file}")
        print("="*80)
        
        return results
    
    def create_summary_report(self, results: List[Dict[str, any]], output_dir: str):
        """Create a summary report with statistics."""
        report_file = os.path.join(output_dir, "summary_report.txt")
        
        # Calculate statistics
        content_types = {}
        total_words = 0
        for result in results:
            content_type = result['content_type']
            content_types[content_type] = content_types.get(content_type, 0) + 1
            total_words += result['word_count']
        
        # Create report content
        report_content = f"""MARX TEXT SUMMARIZER - SUMMARY REPORT
{'='*60}

GENERAL STATISTICS:
- Total files processed: {len(results)}
- Total summary words: {total_words}
- Average words per summary: {total_words / len(results) if results else 0:.1f}

CONTENT TYPE DISTRIBUTION:
"""
        
        for content_type, count in sorted(content_types.items()):
            percentage = (count / len(results)) * 100 if results else 0
            report_content += f"- {content_type.capitalize()}: {count} files ({percentage:.1f}%)\n"
        
        report_content += f"""
TOP KEY CONCEPTS:
"""
        # Count key concepts
        concept_counts = {}
        for result in results:
            for concept in result['key_concepts']:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        # Show top 10 concepts
        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for concept, count in top_concepts:
            report_content += f"- {concept}: {count} occurrences\n"
        
        report_content += f"""
PROCESSING DETAILS:
- Generated by: Marx Text Summarizer
- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Output directory: {output_dir}
- Individual summary files: {len(results)}
- Comprehensive JSON file: comprehensive_summaries.json

{'='*60}
"""
        
        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Created summary report: {report_file}")

def main():
    """Main function to run the final summarizer."""
    print("MARX TEXT SUMMARIZER - BATCH PROCESSING")
    print("="*80)
    
    # Initialize summarizer
    summarizer = FinalMarxTextSummarizer()
    
    # Set output directory to Downloads
    downloads_path = os.path.expanduser("~/Downloads")
    output_dir = os.path.join(downloads_path, "marx_summaries")
    
    print(f"Output directory: {output_dir}")
    print(f"Processing all Marx text files...")
    
    # Process the Marx chapters directory
    directory_path = "."  # Current directory
    results = summarizer.process_directory(directory_path, output_dir)
    
    # Print final statistics
    print(f"\n" + "="*80)
    print("FINAL STATISTICS:")
    print("="*80)
    
    content_types = {}
    for result in results:
        content_type = result['content_type']
        content_types[content_type] = content_types.get(content_type, 0) + 1
    
    for content_type, count in sorted(content_types.items()):
        print(f"{content_type.capitalize()}: {count} files")
    
    print(f"\nAll summary files have been created in: {output_dir}")
    print("Each file has its own individual summary file with detailed analysis.")

if __name__ == "__main__":
    main() 