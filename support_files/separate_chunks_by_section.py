import pdfplumber
from langchain.schema import Document

def extract_sections_by_bold_font(pdf_path):
    current_section = {"title": None, "content": ""}
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            # Get character-level data
            creating_title = False
            prev_y = None
            for char in page.chars:
                text = char.get("text", "")
                font_name = char.get("fontname", "")
                is_bold = "Bold" in font_name  # Adjust based on your PDF's font styles
                current_y = char["top"]

                if is_bold and not(creating_title):
                    # If we encounter bold text and already have a section, close it
                    if current_section["title"]:
                        document = Document(page_content=current_section["content"],
                                            metadata={"page": page_number,
                                                      "section": current_section["title"]})
                        documents.append(document)
                        current_section = {"title": None, "content": ""}
                        prev_y = None

                    # Start a new section
                    current_section["title"] = text
                    creating_title = True
                elif is_bold and creating_title:
                    current_section["title"] += text
                else:
                    # Add content to the current section
                    creating_title = False

                    if prev_y is not None:
                        diff_in_height = abs(prev_y-current_y)
                        if diff_in_height > 9:
                            current_section["content"] += "\n"
                        elif diff_in_height > 0.5:
                            current_section["content"] += " "
                    prev_y = current_y

                    current_section["content"] += text

    # Append the last section, if any
    if current_section["title"]:
        document = Document(page_content=current_section["content"],
                            metadata={"page": page_number,
                                      "section": current_section["title"]})
        documents.append(document)

    return documents

# Specify your PDF file
pdf_path = "your_file.pdf"
sections = extract_sections_by_bold_font('./pdf_files/owner_manual_p283-p300.pdf')

# Print extracted sections
for i, section in enumerate(sections, start=1):
    print(f"\nSection {i}: {section.metadata['section']}")
    print(f"Content:\n{section.page_content}")


