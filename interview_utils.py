import pandas as pd


def load_questions(file_path):
    questions_df = pd.read_csv(file_path)
    return questions_df


def select_question(questions_df, field, question_number):
    field_questions = questions_df[questions_df["Field"] == field].reset_index(drop=True)
    selected = field_questions.iloc[question_number]
    return selected


def check_answer(answer, keywords):
    keyword_list = [word.strip().lower() for word in keywords.split("|")]
    answer_lower = answer.lower()

    matched_words = []
    missing_words = []

    for word in keyword_list:
        if word in answer_lower:
            matched_words.append(word)
        else:
            missing_words.append(word)

    results_df = pd.DataFrame({
        "Key idea": keyword_list,
        "Included": ["Yes" if word in matched_words else "No" for word in keyword_list]
    })

    return {
        "word_count": len(answer.split()),
        "matched_words": matched_words,
        "missing_words": missing_words,
        "results_df": results_df
    }

