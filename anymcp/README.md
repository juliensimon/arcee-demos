# Arcee MCP Examples

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Arcee AnyMCP](https://img.shields.io/badge/Arcee-AnyMCP-orange)](https://arcee.ai)

Minimal test scripts for Arcee's Multi-Cloud Platform (MCP).

## Overview

This directory contains minimal examples and test scripts for working with Arcee AnyMCP, which allows for seamless deployment and management of AI models across different cloud providers.

## Files

- `minimal_test.py`: A minimal test script for basic MCP functionality
- `script.py`: Example script for MCP operations
- `claude_desktop_config.json`: Configuration file for Claude desktop integration

## Requirements

Dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Getting Started

1. Set up your environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

2. Configure your API credentials in the appropriate configuration file.

3. Run the minimal test:
   ```bash
   python minimal_test.py
   ```
