import markdown
from xhtml2pdf import pisa
from pathlib import Path

def convert_to_slides_pdf(md_file, pdf_file):
    """Convert markdown to a Landscape PDF Presentation using xhtml2pdf"""
    
    # Read markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert MD to HTML (basic)
    html_body = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    # Style for Landscape Slides
    # We use @page to define landscape size and margins
    # We use aggressive page breaking for headers to simulate slides
    css_style = """
    <style>
        @page {
            size: a4 landscape;
            margin: 1cm;
            @frame footer_frame {           /* Static Frame */
                -pdf-frame-content: footerContent;
                bottom: 1cm;
                margin-left: 1cm;
                margin-right: 1cm;
                height: 1cm;
            }
        }

        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 24px;
            color: #333;
            line-height: 1.5;
        }

        h1 {
            font-size: 48px;
            color: #2c3e50;
            text-align: center;
            margin-top: 15%;
            margin-bottom: 20px;
        }
        
        /* Subtitle / Author info on title page */
        p {
            margin-bottom: 10px;
        }

        /* Slide Titles (h2) */
        h2 {
            font-size: 36px;
            color: #2980b9;
            border-bottom: 3px solid #2980b9;
            padding-bottom: 10px;
            margin-top: 0;
            page-break-before: always; /* Force new slide for every H2 */
        }

        /* First H2 (Contents) shouldn't break if it follows title immediately, 
           but in our MD structure title is H1. 
           We actually want H2 to always start a page. */

        h3 {
            font-size: 28px;
            color: #16a085;
            margin-top: 20px;
        }

        ul {
            margin-top: 10px;
        }

        li {
            margin-bottom: 15px;
        }
        
        code, pre {
            background-color: #f0f0f0;
            font-family: Courier;
            font-size: 18px;
            padding: 5px;
        }

        /* Hide the manual page breaks from MD since we use CSS basics */
        div {
            display: none; 
        }
        
        #footerContent {
            text-align: right;
            color: #7f8c8d;
            font-size: 12px;
        }
    </style>
    """

    # Footer Content
    footer_html = """
    <div id="footerContent">
        Smart Road Early Warning System - Zeroth Review
    </div>
    """

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        {css_style}
    </head>
    <body>
        {footer_html}
        {html_body}
    </body>
    </html>
    """

    # Generate PDF
    with open(pdf_file, "wb") as pdf_out:
        pisa_status = pisa.CreatePDF(full_html, dest=pdf_out)

    if pisa_status.err:
        print(f"❌ Error creating PDF: {pisa_status.err}")
    else:
        print(f"✅ Presentation PDF created: {pdf_file}")

if __name__ == "__main__":
    md_path = Path("project_presentation.md")
    pdf_path = md_path.with_suffix(".pdf")
    convert_to_slides_pdf(md_path, pdf_path)
