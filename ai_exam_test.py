import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
import matplotlib.pyplot as plt
from langchain_core.messages import AIMessage
import json

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

if not GOOGLE_API_KEY:
    st.error("Google API Key not found. Please check your .env file.")

st.title("AI-Powered Examination Application")
task = st.sidebar.selectbox("Select a Task", ["MCQ Question", "Code Evaluation", "Answer Evaluation"])


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

def MCQ_Generate(keyword, experience):
    research_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    message = [("system", f"Generate ten (10) MCQ questions and their correct answers with four (4) options based on the user's skillset and experience in {keyword}: {experience} in JSON Format."), ("human", "")]
    return research_model.invoke(message)

def extract_qa(llm_response: AIMessage):
    try:
        json_string = llm_response.content.strip().replace('```json', '').replace('```', '').strip()
        qa_list = json.loads(json_string)
        return qa_list
    except json.JSONDecodeError:
        print("JSON Decode Error: Could not parse LLM response as JSON.")
        print("Raw response:", llm_response.content)
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def generate_subjective_questions(keywords, experience):
    """Generates subjective questions using LLM."""
    prompt = f"Generate 5 subjective programming questions for {keywords} at {experience} level in JSON format. Each question should have 'question' and 'expected_answer' keys. Do not include any extra text other than the json output."
    response = llm.invoke(prompt)
    return response


def evaluate_answer(question, student_answer):
    """Evaluates student's answer using LLM."""
    prompt = f"""Evaluate the following student answer for correctness against the programming question:
    Question: {question['question']}
    Expected Answer: {question['expected_answer']}
    Student Answer: {student_answer}
    Provide a percentage score and feedback in JSON format with 'score' and 'feedback' keys. Do not include any extra text other than the json output."""
    response = llm.invoke(prompt)
    try:
        json_string = response.content.strip().replace('```json', '').replace('```', '').strip()
        evaluation = json.loads(json_string)
        return [evaluation]  # Ensure that the evaluation is a list containing a dict.
    except json.JSONDecodeError:
        st.error("JSON Decode Error: Could not parse LLM evaluation response as JSON.")
        print("Raw evaluation response:", response.content)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def generate_programming_questions(keywords, experience):
    """Generates programming questions using LLM."""
    prompt = f"Generate 5 programming questions for {keywords} at {experience} level in JSON format. Each question should have 'question' and 'expected_code' keys. Do not include any extra text other than the json output."
    response = llm.invoke(prompt)
    return response

def evaluate_code(question, student_code):
    """Evaluates student's code using LLM."""
    prompt = f"""Evaluate the following student code for correctness against the programming question:
    Question: {question['question']}
    Expected Code: {question['expected_code']}
    Student Code: {student_code}

    Provide a strict percentage score (0-100) reflecting the accuracy and functionality of the student code compared to the expected code. Provide feedback that explains the score. If the student code is entirely incorrect, the score should be 0. If it perfectly matches, it should be 100. Provide the result in JSON format with 'score' and 'feedback' keys. Do not include any extra text other than the json output."""
    response = llm.invoke(prompt)
    try:
        json_string = response.content.strip().replace('```json', '').replace('```', '').strip()
        evaluation = json.loads(json_string)
        return [evaluation] # Ensure that the evaluation is a list containing a dict.
    except json.JSONDecodeError:
        st.error("JSON Decode Error: Could not parse LLM evaluation response as JSON.")
        print("Raw evaluation response:", response.content)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None



if task == "MCQ Question":
    st.title("Welcome to MCQ Test")
    query_params = st.experimental_get_query_params()
    keywords = query_params.get("keywords", ["python"])[0].split(",")
    experience = query_params.get("experience", ["3"])[0]
    st.write("Keywords:", keywords[0])
    st.write("Experience:", experience)

    if 'exam_started' not in st.session_state:
        st.session_state.exam_started = False
    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = None
    if 'questions' not in st.session_state:
        st.session_state.questions = None
    if 'student_answers' not in st.session_state:
        st.session_state.student_answers = None

    if not st.session_state.exam_started:
        if st.button("Start Exam"):
            st.session_state.exam_started = True
            st.session_state.llm_response = MCQ_Generate(keywords, experience)
            st.session_state.questions = extract_qa(AIMessage(content=st.session_state.llm_response.content))
            if st.session_state.questions:
                st.session_state.student_answers = [None] * len(st.session_state.questions)
            else:
                st.error("Failed to generate questions. Please check the keywords and experience.")
                st.stop()

    if st.session_state.exam_started:
        if st.session_state.questions:
            def calculate_score(student_answers):
                correct_answers = 0
                for i, answer in enumerate(student_answers):
                    if answer is not None and i < len(st.session_state.questions):
                        if answer == st.session_state.questions[i]['answer']:
                            correct_answers += 1
                return correct_answers

            for i, q in enumerate(st.session_state.questions):
                st.subheader(q['question'])
                # Find the index of the selected answer
                selected_index = None
                if st.session_state.student_answers and st.session_state.student_answers[i] in q['options']:
                    selected_index = q['options'].index(st.session_state.student_answers[i])

                answer = st.radio(
                    "Choose one option",
                    q['options'],
                    index=selected_index, # Use the numerical index
                    key=f"{q['question']}_{i}"
                )
                if st.session_state.student_answers:
                    st.session_state.student_answers[i] = answer

            if st.button("Submit Test"):
                if st.session_state.student_answers and None in st.session_state.student_answers:
                    st.warning("Please answer all questions before submitting.")
                else:
                    score = calculate_score(st.session_state.student_answers)
                    total_questions = len(st.session_state.questions)
                    st.write(f"Your score is: {score}/{total_questions}")

                    correct = sum([1 for i in range(total_questions) if st.session_state.student_answers[i] == st.session_state.questions[i]['answer']])
                    incorrect = total_questions - correct

                    fig, ax = plt.subplots()
                    ax.pie([correct, incorrect], labels=['Correct', 'Incorrect'], autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)

                    st.session_state.exam_started = False
                    st.session_state.questions = None
                    st.session_state.student_answers = None
                    st.session_state.llm_response = None
        else:
            if st.session_state.exam_started:
                st.write("Questions were not properly generated.")


elif task == "Answer Evaluation":
    # Streamlit UI
    st.title("Subjective Programming Question Test")

    # Default values
    keywords = "Python"
    experience = "3"

    # Get query parameters from URL
    query_params = st.experimental_get_query_params()
    if "keywords" in query_params:
        keywords = query_params["keywords"][0]
    if "experience" in query_params:
        experience = query_params["experience"][0]

    if 'questions' not in st.session_state:
        st.session_state.questions = None
        st.session_state.student_answers = {}
        st.session_state.evaluations = {}
        st.session_state.current_question_index = 0
        st.session_state.exam_started = False
        st.session_state.evaluation_done = False

    if st.button("Start Exam") and not st.session_state.exam_started:
        response = generate_subjective_questions(keywords, experience)
        if response:
            questions = extract_qa(AIMessage(content=response.content))
            if questions:
                st.session_state.questions = questions
                st.session_state.student_answers = {i: "" for i in range(len(questions))}
                st.session_state.evaluations = {i: None for i in range(len(questions))}
                st.session_state.current_question_index = 0
                st.session_state.exam_started = True
                st.session_state.evaluation_done = False
            else:
                st.error("Failed to parse questions from LLM response.")

    if st.session_state.exam_started:
        if st.session_state.current_question_index < len(st.session_state.questions):
            current_question = st.session_state.questions[st.session_state.current_question_index]
            st.subheader(f"Question {st.session_state.current_question_index + 1}:")
            st.write(current_question["question"])
            student_answer = st.text_area(f"Your Answer (Question {st.session_state.current_question_index + 1})",
                                            value=st.session_state.student_answers.get(st.session_state.current_question_index, ""),
                                            key=f"answer_{st.session_state.current_question_index}")
            st.session_state.student_answers[st.session_state.current_question_index] = student_answer

            if st.button("Submit and Next"):
                evaluation = evaluate_answer(current_question, student_answer)
                st.session_state.evaluations[st.session_state.current_question_index] = evaluation

                if st.session_state.current_question_index + 1 < len(st.session_state.questions):
                    st.session_state.current_question_index += 1
                    st.rerun()
                else:
                    st.session_state.exam_started = False
                    st.session_state.evaluation_done = True
                    st.rerun()

    if st.session_state.evaluation_done:
        st.subheader("Evaluations:")
        for i, question in enumerate(st.session_state.questions):
            st.subheader(f"Evaluation for Question {i + 1}:")
            evaluation = st.session_state.evaluations.get(i)
            if evaluation and isinstance(evaluation, list) and len(evaluation) > 0 and isinstance(evaluation[0], dict) and 'score' in evaluation[0] and 'feedback' in evaluation[0]:
                st.write(f"Score: {evaluation[0]['score']}%")
                st.write(f"Feedback: {evaluation[0]['feedback']}")
            else:
                st.write("Evaluation not available or invalid format.")

    if not st.session_state.exam_started and not st.session_state.evaluation_done:
        st.write(f"Current keyword: {keywords}, Experience level: {experience}")
        st.write("Click 'Start Exam' to begin.")

elif task == "Code Evaluation":
    # Streamlit UI
    st.title("Welcome to Coding/Programming Test")

    # Default values
    keywords = "Python"
    experience = "3"

    # Get query parameters from URL
    query_params = st.experimental_get_query_params()
    if "keywords" in query_params:
        keywords = query_params["keywords"][0]
    if "experience" in query_params:
        experience = query_params["experience"][0]

    if 'questions' not in st.session_state:
        st.session_state.questions = None
        st.session_state.student_codes = {}
        st.session_state.evaluations = {}
        st.session_state.current_question_index = 0
        st.session_state.exam_started = False
        st.session_state.evaluation_done = False

    if st.button("Start Exam") and not st.session_state.exam_started:
        st.session_state.questions = None  # Reset questions before generating new ones.
        st.session_state.student_codes = {}
        st.session_state.evaluations = {}
        st.session_state.current_question_index = 0
        st.session_state.exam_started = True
        st.session_state.evaluation_done = False
        response = generate_programming_questions(keywords, experience)
        if response:
            questions = extract_qa(AIMessage(content=response.content))
            if questions:
                st.session_state.questions = questions
                st.session_state.student_codes = {i: "" for i in range(len(questions))}
                st.session_state.evaluations = {i: None for i in range(len(questions))}
            else:
                st.error("Failed to parse questions from LLM response.")

    if st.session_state.exam_started:
        if st.session_state.current_question_index < len(st.session_state.questions):
            current_question = st.session_state.questions[st.session_state.current_question_index]
            st.subheader(f"Question {st.session_state.current_question_index + 1}:")
            st.write(current_question["question"])
            student_code = st.text_area(f"Your Code (Question {st.session_state.current_question_index + 1})",
                                            value=st.session_state.student_codes.get(st.session_state.current_question_index, ""),
                                            key=f"code_{st.session_state.current_question_index}")
            st.session_state.student_codes[st.session_state.current_question_index] = student_code

            if st.button("Submit and Next"):
                evaluation = evaluate_code(current_question, student_code)
                st.session_state.evaluations[st.session_state.current_question_index] = evaluation

                if st.session_state.current_question_index + 1 < len(st.session_state.questions):
                    st.session_state.current_question_index += 1
                    st.rerun()
                else:
                    st.session_state.exam_started = False
                    st.session_state.evaluation_done = True
                    st.rerun()

    if st.session_state.evaluation_done:
        st.subheader("Evaluations:")
        for i, question in enumerate(st.session_state.questions):
            st.subheader(f"Evaluation for Question {i + 1}:")
            evaluation = st.session_state.evaluations.get(i)
            if evaluation and isinstance(evaluation, list) and len(evaluation) > 0 and isinstance(evaluation[0], dict) and 'score' in evaluation[0] and 'feedback' in evaluation[0]:
                st.write(f"Score: {evaluation[0]['score']}%")
                st.write(f"Feedback: {evaluation[0]['feedback']}")
            else:
                st.write("Evaluation not available or invalid format.")

    if not st.session_state.exam_started and not st.session_state.evaluation_done:
        st.write(f"Current keyword: {keywords}, Experience level: {experience}")
        #st.write("Change the url parameter to change keyword and experience level. Click 'Start Exam' to begin.")
