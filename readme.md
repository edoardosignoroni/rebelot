# ANNOTATED CORPUS OF LOMBARD CODE-MIXING AND SWITCHING

## ANNOTATION

You can use json_to_vert.py to convert the data from .vert to .jsonl, and viceversa

    python json_to_vert.py to-jsonl -i <input_file> -o <output_file>
    python json_to_vert.py to-vert -i <input_file> -o <output_file>

## INFO ON THE CORPUS

**books.txt**: from a portion of transcriptions of interviews on oral tradition and stories in the Bergamo area


**proverbs.txt**: from a website collecting Brescian proverbs and their explanations


**news.txt**: articles on Brescian from Giornale di Brescia


**socials.txt**: instagram posts from @bresciadice, code-mixed humoristic explanations of common idioms from Brescia


## STATISTICS

Lines: 704

Words: 79611

Characters: 526882

Empty lines: 0

Vocabulary size: 23146

Average line length (words): 113.08

Average word length (chars): 5.63

Top 10 most used words: [('di', 1822), ('che', 1501), ('e', 1428), ('la', 1264), ('il', 1263), ('a', 1107), ('in', 876), ('un', 808), ('è', 716), ('/', 641)]

Least used 10 words: [('#<lmo>GhétVestCheRobe</lmo>', 1), ('capitale', 1), ('robe??"', 1), ('</eng><lmo>"(Pota)', 1), ('Mattarella,', 1), ('Mr.</eng>', 1), ('fireworks,', 1), ('Party,', 1), ('Opening', 1), ('stuff?,', 1)]

Total <eng> words: 5980

Average <eng> words per line: 8.49 (min: 1, max: 83)

Total <ita> words: 63206

Average <ita> words per line: 89.78 (min: 0, max: 661)

Total <lmo> words: 11658

Average <lmo> words per line: 16.56 (min: 1, max: 612)

## LICENSE and COPYRIGHT

The data is provided in a de-contextualized, tokenized format (.vert). The work is used for non-expressive, scientific purposes. The original text has been tokenized, shuffled, and annotated. All rights to the underlying original strings remain with the original creators.  Use of this data is grounded in the Text and Data Mining (TDM) exceptions for research (Art. 3, EU Directive 2019/790; Art. 70-ter Italian Law 633/1941).

This corpus is released strictly for non-commercial scientific research purposes under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike) license. 
