# from sentence_transformers import SentenceTransformer

# from Bert import extract_keywords
# from ReadDoc import read_doc

# # BERT modelini yüklə
# model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# # Hər bir faylın mətni üçün açar sözləri çıxarın
# # keywords_list = []
# # for file_path in file_paths:
# #     text = read_doc(file_path)
# #     keywords = extract_keywords(text)
# #     keywords_list.append(keywords)

# # Açar sözlərin vektorlarını yaradın
# keyword_vectors = [model.encode(keywords) for keywords in keywords_list]

# # Vektorları birləşdirin (istəyə görə özəlləşdirilə bilər)
# import numpy as np
# keyword_vectors_combined = [np.mean(keywords, axis=0) for keywords in keyword_vectors]

# # İndi hər açar söz vektorunun təmsil etdiyi mövzu və ya faylın xüsusiyyətlərini ehtiva edən bir vektor olacaq
