import fitz  # PyMuPDF
import os

def convert_pdfs_in_data_folder():
    # Define the path to the data folder
    data_folder = "data"
    
    # Ensure the data folder exists
    if not os.path.exists(data_folder):
        print("The 'data' folder does not exist.")
        return
    
    # Loop through all files in the 'data' folder
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".pdf"):  # Check if the file is a PDF
            pdf_path = os.path.join(data_folder, filename)
            txt_filename = filename.replace(".pdf", ".txt")
            txt_path = os.path.join(data_folder, txt_filename)
            
            # Open the PDF file
            doc = fitz.open(pdf_path)
            
            # Initialize an empty string to store the extracted text
            text = ""
            
            # Loop through each page and extract text
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
            
            # Write the extracted text to a .txt file
            with open(txt_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(text)

            print(f"Converted {filename} to {txt_filename}")

# Run the conversion
convert_pdfs_in_data_folder()
