import zipfile
with zipfile.ZipFile('Juliet_Test_Suite_v1.3_for_C_Cpp.zip', 'r') as zip_ref:
    zip_ref.extractall('.')