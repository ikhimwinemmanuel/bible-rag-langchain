from rag_qa import answer_question

def run_sanity_tests():
    questions = [
        "Where in the Bible does it mention the creation of man?",
        "Who built the ark?",
        "What is the greatest commandment?",
    ]

    for q in questions:
        print(f"\n🔍 QUESTION: {q}")
        try:
            answer = answer_question(q)
            print(f"ANSWER:\n{answer}")
        except Exception as e:
            print(f"❌ Error answering question: {e}")

if __name__ == "__main__":
    run_sanity_tests()
