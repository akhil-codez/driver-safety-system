"""
Convert markdown to styled HTML for browser-based PDF export
"""
import markdown
from pathlib import Path

def convert_md_to_html(md_file, html_file):
    """Convert markdown file to styled HTML"""
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_body = markdown.markdown(
        md_content,
        extensions=['tables', 'fenced_code', 'codehilite', 'nl2br']
    )
    
    # Create full HTML with styling
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Driver Safety System - Tech Stack & Models</title>
    <style>
        @media print {
            body { margin: 0; }
            .no-print { display: none; }
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #333;
            background: #fff;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 4px solid #3498db;
            padding-bottom: 12px;
            margin-top: 40px;
            font-size: 2.5em;
        }
        
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin-top: 35px;
            font-size: 2em;
        }
        
        h3 {
            color: #555;
            margin-top: 25px;
            font-size: 1.5em;
        }
        
        h4 {
            color: #666;
            margin-top: 20px;
        }
        
        code {
            background-color: #f4f4f4;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }
        
        pre {
            background-color: #f8f8f8;
            border: 1px solid #ddd;
            border-left: 4px solid #3498db;
            border-radius: 5px;
            padding: 20px;
            overflow-x: auto;
            line-height: 1.4;
        }
        
        pre code {
            background: none;
            padding: 0;
            color: #333;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 25px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        th, td {
            border: 1px solid #ddd;
            padding: 14px;
            text-align: left;
        }
        
        th {
            background-color: #3498db;
            color: white;
            font-weight: bold;
            text-transform: uppercase;
            font-size: 0.9em;
        }
        
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        blockquote {
            border-left: 5px solid #3498db;
            padding-left: 20px;
            margin-left: 0;
            color: #555;
            font-style: italic;
            background: #f9f9f9;
            padding: 15px 20px;
            border-radius: 4px;
        }
        
        ul, ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        
        li {
            margin: 8px 0;
        }
        
        hr {
            border: none;
            border-top: 2px solid #ecf0f1;
            margin: 40px 0;
        }
        
        .print-button {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
            z-index: 1000;
        }
        
        .print-button:hover {
            background: #2980b9;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.3);
        }
        
        .header-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        .header-info h1 {
            color: white;
            border: none;
            margin: 0;
        }
        
        .header-info p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        
        strong {
            color: #2c3e50;
        }
        
        em {
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <button class="print-button no-print" onclick="window.print()">🖨️ Save as PDF</button>
    
    <div class="header-info">
        <h1>📚 Driver Safety System</h1>
        <p><strong>Tech Stack & ML Models Documentation</strong></p>
        <p>Complete technical reference for Python training pipeline and Android deployment</p>
    </div>
    
    {CONTENT}
    
    <script>
        // Add smooth scroll behavior
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
    </script>
</body>
</html>"""
    
    # Replace content placeholder
    final_html = html_template.replace('{CONTENT}', html_body)
    
    # Write HTML file
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"✅ HTML created successfully: {html_file}")
    print(f"\n📖 Instructions:")
    print(f"1. Open the HTML file in your browser")
    print(f"2. Click the 'Save as PDF' button (or press Ctrl+P)")
    print(f"3. Select 'Save as PDF' as the printer")
    print(f"4. Click 'Save' and choose your location")

if __name__ == "__main__":
    import sys
    md_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tech_stack_explanation.md")
    html_file = md_file.with_suffix('.html')
    
    convert_md_to_html(md_file, html_file)
