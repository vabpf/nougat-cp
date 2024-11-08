from fpdf import FPDF
from PIL import Image
import base64

def create_pdf(image_path, output_pdf):
    pdf = FPDF()
    pdf.add_page()

    # Add image
    pdf.image(image_path, x=10, y=8, w=100)

    pdf.output(output_pdf)

def create_html(text_path, output_html):

    # Read text
    with open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
    </head>
    <body>
        <div class="ltx_document">
            <div class="ltx_title_section">
                <h1 class="ltx_title">Document Title</h1>
            </div>
            <div class="ltx_section">
                <p class="ltx_p">{text}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Write HTML to file
    with open(output_html, 'w', encoding='utf-8') as file:
        file.write(html_content)

# Example usage
if __name__ == "__main__":
    create_pdf(r"D:\Data\vietnamese_ocr_data\InkData_line_processed\20140603_0003_BCCTC_tg_0_0.png", 'test/data/20140603_0003_BCCTC_tg_0_0.pdf')
    create_html(r"D:\Data\vietnamese_ocr_data\InkData_line_processed\20140603_0003_BCCTC_tg_0_0.txt", 'test/data/20140603_0003_BCCTC_tg_0_0.html')