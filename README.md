# Question Answering System
We built a Question Answer System using BERT Transformer. Based on our benchmark dataset that we designed for a specific task, we evaluated it at a **TBD**.

## What can it do?
1. This QA System is topic agnostic - there is no inbuilt context. Depending on the context you feed it, you can ask questions about that (there is a little bit of structure for our task listed below).
2. It selects the top 3 documents in a corpus and outputs the answer with the highest confidence score.
3. It checks the spelling (Damerau-Levenshtein distance) and grammar (t5-base-grammar-correction) of the question before feeding it to BERT.
4. It does remember context history as long as you are talking about one object, if you switch objects and then refer to the new object's history, it'll get confused :( - you can check some of our older issues for more information.
5. If you ask about something that it does not have in context, it will respond with "Unable to answer" (most of the time).

## Corpus used
We used a specific corpus for our task, however it can be used according to your needs as well. The current structure is:
- prompted to choose from 3 different corpuses
- all further questions based on selected corpus
- once you decide to stop asking questions, the system prompts whether you want to learn more
- if you decide yes, then three more choices are offerred
- if no, conversation ends

## Instructions
1. Install the following - 
- `pip install transformers` (for BERT for QA)
- `pip install happytransformer` (for grammar checking which uses t5 model)
- `pip install symspellpy` (for spell checking)

2. Feed your corpus as a dataframe to `current_phase_df`
3. Use `begin_conversation` method in `question_answer_bot.py` to get started!
