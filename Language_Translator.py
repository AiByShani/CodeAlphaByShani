import googletrans
from googletrans import Translator

translator = Translator()
# Input data from user

Text=input("Enter text: ")
print("Chose Your Language According to keywords for each Language:")
print(googletrans.LANGUAGES)
Src_Lang=input("Entrer Input Language: ")
Des_Lang=input("Entrer Output Language: ")


# Perform Translation

T_Text=translator.translate(Text, src=Src_Lang, dest=Des_Lang)
print(f"Translate from {Src_Lang} to {Des_Lang} ")
print(f"{googletrans.LANGUAGES[Src_Lang]} : {Text}")
print(f"{googletrans.LANGUAGES[Des_Lang]} : {T_Text.text}")