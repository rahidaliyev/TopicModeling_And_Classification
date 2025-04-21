import win32com.client

def read_doc(file_path):
    # Microsoft Word tətbiqini açırıq
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False  # Word tətbiqinin görünməsini istəyirikmi? False olaraq buraxırıq
    # DOC faylını açırıq
    doc = word.Documents.Open(file_path)
    
    # Faylın mətnini oxuyuruq
    full_text = doc.Content.Text
    
    # Faylı bağlayırıq
    doc.Close()
    # Word tətbiqini bağlayırıq
    word.Quit()

    return full_text

# DOC faylınızın yolu
# file_path = r"C:\\Users\\rahid\\Documents\\projects\\advanced_machine_learning\\BERTZ\\text_files\\Alcohol.doc"

# # Faylı oxuyuruq
# text = read_doc(file_path)
# print(text)
