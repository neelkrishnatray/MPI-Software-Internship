import os
import pypandoc

def save_pdf(markdown_text):

    try:
        os.makedirs("outputs", exist_ok = True) #Ordner Outputs wird erstellt, falls er nicht funktioniert

        pypandoc.convert_text(
            markdown_text,
            "pdf",
            format = "md",
            outputfile = "outputs/final_report.pdf",
            extra_args = ["--standalone", "--toc"]
        )

        print("PDF erstellt.")

    except Exception as e:
        print("PDF Generator Fehler: ", e)
