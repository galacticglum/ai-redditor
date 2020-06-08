#!/bin/bash
# 
# A script for scraping posts from r/writingprompts.
# Usage: ./scrape_writingprompts.sh <credentials> <output_file>

if [ '$0' != '' && '$1' != '' ];  then
    python ai-redditor\\reddit_scraper.py writingprompts $1 -c $0 --query WP --repair --top-level-comments --title-filter "^(\s*\[WP\])(?!.*\[OT\])" --comment-filter "^(?!\[OT\])"
else
    echo 'Invalid arguments. See usage for documentation on arguments.'
fi