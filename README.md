
# Marx Text Files Summarizer

A comprehensive tool for generating accurate descriptive summaries (~50 words) for Marx text files, with intelligent content type detection and scholarly analysis.

## Project Overview

This project processes a large collection of Marx's texts including poetry, philosophy, economics, political analysis, and correspondence. It generates concise, accurate summaries while preserving the original file structure and providing detailed content analysis.

## Features

### **Content Type**
- **Economics**: Capital, commodities, value theory, political economy
- **Philosophy**: Hegelian dialectics, materialism, critique of religion
- **Poetry**: Early verses, ballads, dedications
- **Politics**: Class struggle, revolution, state analysis
- **Correspondence**: Letters, personal communications
- **Unknown**: Fallback classification for mixed content

### **Advanced Analysis**
- Key concept extraction (Marx, Engels, Capital, Revolution, etc.)
- Content length analysis
- Writing style identification
- Historical context preservation
- 
## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key (optional, for enhanced summaries)

### Setup
```bash
# Clone or download the project
cd marx_chapters_txt_extracted

# Create virtual environment
python -m venv marx_env
source marx_env/bin/activate  # On Windows: marx_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (optional)
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage
```bash
# Process all files in current directory
python final_summarizer.py
```

### Advanced Usage
```python
from final_summarizer import FinalMarxTextSummarizer

# Initialize summarizer
summarizer = FinalMarxTextSummarizer(api_key="your-openai-key")

# Process specific directory
results = summarizer.process_directory("./marx_chapters_v1", "~/Downloads/output")
```

### Demo Script
```bash
# Run OpenAI API demonstration
python demo_with_openai.py
```

## Output Structure

### Individual Summary Files
Each processed file generates a corresponding `_summary.txt` file:

```
MARX TEXT SUMMARY
==================================================

ORIGINAL FILE: ./marx_chapters_v1/v1_Book_of_Verse_to_Father_01_Contents.txt
TITLE: v1_Book_of_Verse_to_Father_01_Contents
CONTENT TYPE: correspondence
KEY CONCEPTS: marx, hegel
WORD COUNT: 22

SUMMARY:
Book of Verse to Father 01 Contents: Personal letter from Marx to his father (1837) 
discussing his studies, poetry, and philosophical development.

ANALYSIS:
- Content Type: correspondence
- Length: 1273 characters
- Has Poetry: False
- Has Philosophy: False
- Has Economics: False
- Has Politics: False
- Has Correspondence: True

GENERATED BY: Marx Text Summarizer
DATE: 2025-06-17 21:16:14
```

### Directory Organization
```
marx_summaries/
├── marx_chapters_v1/
│   ├── v1_Book_of_Verse_to_Father_01_Contents_summary.txt
│   └── ...
├── marx_chapters_v35/
│   ├── v35_K_Marx_Chapter_1_Commodities_summary.txt
│   └── ...
├── comprehensive_summaries.json
└── summary_report.txt
```

## Example Outputs

### Economics Text
```
Chapter 1 Commodities: Marx's analysis of commodities, use value, exchange value, 
and the labor theory of value from Capital.
```

### Poetry Text
```
Book of Verse to Father: Poetry collection by Marx dedicated to his father in 1837, 
containing verses and ballads reflecting his early philosophical development.
```

### Political Analysis
```
Manifesto of the Communist Party: Revolutionary political text by Marx and Engels 
outlining class struggle, historical materialism, and communist principles.
```

### Correspondence
```
Letter to Father: Personal correspondence from Marx to his father discussing 
his studies, poetry, and philosophical development during his university years.
```

## Technical Design

### Content Type Detection Algorithm
1. **Priority-based Classification**:
   - Economics (highest priority)
   - Correspondence/Letters
   - Poetry
   - Philosophy
   - Politics
   - Unknown (fallback)

2. **Keyword Analysis**:
   - Title-based detection
   - Content-based detection
   - Contextual analysis

### Summary Generation Strategy
- **OpenAI API**: Scholarly, accurate summaries with historical context
- **Fallback System**: Rule-based extraction with key concept identification
- **Length Control**: ~50 words target with flexibility
- **Content Preservation**: Maintains original meaning and context

### Error Handling
- File reading errors
- API connection issues
- Content parsing failures
- Rate limiting protection

## Performance Statistics

### Processing Results
- **Total Files Processed**: 3,371
- **Success Rate**: 99.9%
- **Average Summary Length**: 17.3 words
- **Processing Time**: ~2 hours (with API delays)

### Content Distribution
- **Politics**: 1,334 files (39.6%)
- **Poetry**: 592 files (17.6%)
- **Unknown**: 712 files (21.1%)
- **Correspondence**: 387 files (11.5%)
- **Economics**: 187 files (5.5%)
- **Philosophy**: 159 files (4.7%)

### Top Key Concepts
- Marx: 2,906 occurrences
- Engels: 2,231 occurrences
- State: 1,319 occurrences
- Capital: 969 occurrences
- Revolution: 894 occurrences


## License

This project is designed for academic and research purposes. Please ensure compliance with relevant copyright and usage restrictions for Marx's texts.


=======
## This is a repository for my summer RA job

