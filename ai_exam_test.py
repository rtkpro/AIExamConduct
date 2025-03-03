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

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
llm_eval = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

def MCQ_Generate(keyword, experience):
    message = prompt = [
    (
        "system",
        f"""
        You are an expert MCQ (Multiple Choice Question) generator. Your task is to create ten (10) high-quality MCQ questions based on the user's provided skillset and experience. 

        **Constraints:**
        * Each question must have exactly four (4) distinct options (A, B, C, D).
        * One and only one option must be the correct answer.
        * The questions should be relevant to the skillset: '{keyword}' and experience level: '{experience}'.
        * The questions should be challenging but fair, suitable for someone with the specified experience.
        * Avoid overly complex or ambiguous questions.
        * Format your response strictly as a JSON list of dictionaries.
        * Each dictionary must have the following keys: "question", "options" (a list of four strings), and "answer" (the correct answer string).
        * Ensure the JSON is valid and parsable.
        * Do not include any introductory or explanatory text outside of the JSON.
        * Do not include any markdown code blocks, just the raw json.

        **Example JSON Output Format:**
        [
            {{
                "question": "What is the capital of France?",
                "options": ["London", "Paris", "Berlin", "Madrid"],
                "answer": "Paris"
            }},
            {{
                "question": "What is 2 + 2?",
                "options": ["3", "4", "5", "6"],
                "answer": "4"
            }},
            // ... 8 more questions ...
        ]

        **User Skillset and Experience:**
        Skillset: {keyword}
        Experience: {experience}
        """
    ),
    ("human", "")
]
    return llm.invoke(message)

def extract_qa(llm_response: AIMessage):
    try:
        if not llm_response or not llm_response.content.strip():
            return None
        json_string = llm_response.content.strip()
        try:
            qa_list = json.loads(json_string)
            return qa_list
        except json.JSONDecodeError as e:
            try:
                json_string = json_string.replace("\n", "").replace("```json","").replace("```","")
                qa_list = json.loads(json_string)
                return qa_list
            except json.JSONDecodeError as e2:
                return None
    except Exception as e:
        return None

def generate_subjective_questions(keywords, experience):
    """Generates subjective questions using LLM."""
    prompt = f"""
    You are an expert programming question generator. Your task is to create 5 subjective programming questions tailored for a programmer with {experience} level experience in {keywords}.

    **Constraints:**
    * Generate exactly 5 subjective programming questions.
    * Each question should be designed to assess the programmer's understanding and practical skills in {keywords}.
    * Provide an 'expected_answer' for each question, representing a plausible and correct solution or approach.
    * Format your response strictly as a JSON list of dictionaries.
    * Each dictionary must contain the keys "question" and "expected_answer".
    * Ensure the JSON is valid and parsable.
    * Do not include any introductory or explanatory text outside of the JSON.
    * Do not include any markdown code blocks, just the raw json.

    **Example JSON Output Format:**
    [
        {{
            "question": "Explain the concept of [Specific {keywords} concept] and provide an example.",
            "expected_answer": "A detailed explanation of [Specific {keywords} concept] with code example."
        }},
        {{
            "question": "Describe a scenario where you would use [Another specific {keywords} concept] and why.",
            "expected_answer": "A description of a specific use case and the reasoning behind using [Another specific {keywords} concept]."
        }},
        // ... 3 more questions ...
    ]

    **Programming Skillset and Experience:**
    Skillset: {keywords}
    Experience: {experience}

    Generate the programming questions.
    """
    response = llm.invoke(prompt)
    return response

def evaluate_answer(question, student_answer):
    """Evaluates student's answer using LLM with advanced prompt engineering."""
    prompt = f"""
    You are an expert programming exam evaluator. Your task is to accurately assess the student's answer against the expected answer for the given programming question.

    **Question:** {question['question']}
    **Expected Answer:** {question['expected_answer']}
    **Student Answer:** {student_answer}

    **Evaluation Criteria:**
    1. Assess the correctness and completeness of the student's answer.
    2. Evaluate the student's understanding of the concepts involved.
    3. Consider the clarity and conciseness of the student's response.
    4. Provide a numerical score between 0 and 100, where 100 represents a perfect answer.
    5. Provide detailed feedback explaining the score and highlighting areas of strength and weakness.
    6. **if the student Answer is *empty/None/blank* cosider score is 0 and provide feedback accordingly.**

    **Constraints:**
    * Provide a JSON response with the keys "score" (integer) and "feedback" (string).
    * Ensure the JSON is valid and parsable.
    * Do not include any introductory or explanatory text outside of the JSON.

    **Example JSON Output Format:**
    {{
        "score": 85,
        "feedback": "The student demonstrated a good understanding of the concept. The code example was mostly correct, but some minor improvements could be made. Overall, a strong answer.",
        "score": 0,
        "feedback": "The student demonstrated a not understanding of the concept. The code example was incorrect."
    }}

    Evaluate the student's answer and provide the JSON response.
    """
    response = llm_eval.invoke(prompt)
    try:
        json_string = response.content.strip().replace('```json', '').replace('```', '').strip()
        evaluation = json.loads(json_string)
        return [evaluation]
    except json.JSONDecodeError:
        st.error("JSON Decode Error: Could not parse LLM evaluation response as JSON.")
        print("Raw evaluation response:", response.content)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


def generate_programming_questions(keywords, experience):
    """Generates programming questions using LLM."""
    prompt = f"""
    You are an expert programming coding question generator. Your task is to create five (5) coding questions tailored for a programmer with {experience} level experience in {keywords}.

    **Constraints:**
    * Generate exactly 5 coding questions.
    * Each question should assess the programmer's understanding and practical skills in {keywords}.
    * The questions should vary in difficulty, testing fundamental to advanced concepts.
    * Provide an 'expected_answer' for each question, representing a plausible and correct solution or approach.
    * Format your response strictly as a JSON list of dictionaries.
    * Each dictionary must contain the keys "question" and "expected_answer". 
    * Ensure the JSON is valid and parsable.
    * Do not include any introductory or explanatory text outside of the JSON.
    * Do not include any markdown code blocks, just the raw JSON.

    **Programming Skillset and Experience:**
    Skillset: {keywords}
    Experience: {experience}

    Generate the coding questions.
    """
    response = llm.invoke(prompt)
    return response

def evaluate_code(question, student_code):
    """Evaluates student's code using LLM."""
    prompt = f"""
    You are an expert programming exam evaluator. Your task is to evaluate the student's code based on the given programming question and expected answer.

    **Question:** {question['question']}
    **Expected Answer:** {question['expected_answer']}
    **Student Answer:** {student_code}

    **Evaluation Criteria:**
    1. **Correctness:** Assess if the student's answer behaves as expected based on the question and expected answer. Check if the code solves the problem and if any errors are present in the student's code.
    2. **Completeness:** Evaluate whether the student has covered all the requirements of the question. Ensure that all parts of the problem are addressed in the code or the provided explanation.
    3. **Clarity and Conciseness:** Check if the student's code is easy to read, follows best practices, and avoids unnecessary complexity or redundancy.
    4. **Understanding of the Concept:** Evaluate if the student used appropriate techniques, algorithms, and approaches to solve the problem. Assess the depth of their understanding based on the solution.
    5. **Edge Case Handling:** Consider if the student's code handles edge cases or provides the correct solution in unexpected or extreme conditions.
    6. Provide a **numerical score** between 0 and 100, where 100 represents a perfect answer. Consider penalizing for errors, incomplete answers, or incorrect approaches.

    **Constraints:**
    - If the **student's answer is empty/None/blank**, consider the score to be 0 and provide feedback explaining the lack of response or effort.
    - Provide a **JSON response** containing the keys:
        - "score" (integer): The numerical score between 0 and 100.
        - "feedback" (string): Detailed feedback explaining the score, areas of strength/weakness, and specific comments on what needs improvement.
    - Ensure the **JSON** is valid and parsable.
    - Do not include any introductory or explanatory text outside of the JSON.
    - Do not include any markdown code blocks, just the raw JSON.

    **Example JSON Output Format:**

    1. **For a score of 0:**
        "score": 0,
        "feedback": "The student's answer is empty or contains no relevant code. Please provide a solution that addresses the problem requirements."

    2. **For a score of 50:**
        "score": 50,
        "feedback": "The student attempted to solve the problem but missed key elements. The logic is partially correct, but errors are present that prevent the code from working as expected. Further attention is needed to handle edge cases and improve the overall solution."

    3. **For a score of 85:**
        "score": 85,
        "feedback": "The student demonstrated a strong understanding of the concept. The solution is mostly correct but could benefit from optimization in certain areas. Edge case handling could be improved for a more robust solution."
    
    Evaluate the student's answer/code and provide the JSON response.
    """

    response = llm_eval.invoke(prompt)
    try:
        json_string = response.content.strip().replace('```json', '').replace('```', '').strip()
        evaluation = json.loads(json_string)
        return evaluation
    except json.JSONDecodeError:
        st.error("JSON Decode Error: Could not parse LLM evaluation response as JSON.")
        print("Raw evaluation response:", response.content)
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

st.title("AI-Powered Examination Application")
task = st.sidebar.selectbox("Select a Task", ["MCQ Question", "Code Evaluation", "Answer Evaluation"])


if task == "MCQ Question":
    st.title("welcome to MCQ Test")

    keywords = "Python"
    experience = "3 years of experience"

    query_params = st.query_params
    if "keywords" in query_params:
        keywords = query_params["keywords"]
    if "experience" in query_params:
        try:
            experience = int(query_params["experience"])
            experience = str(experience) + " years of experience"
        except ValueError:
            st.error("Invalid experience value in URL. Must be an integer.")
            st.stop()

    if 'exam_started' not in st.session_state:
        st.session_state.exam_started = False
    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = None
    if 'questions' not in st.session_state:
        st.session_state.questions = None
    if 'student_answers' not in st.session_state:
        st.session_state.student_answers = None
    if 'evaluation_done' not in st.session_state:
        st.session_state.evaluation_done = False

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
                if isinstance(q, dict): #Check if q is a dictionary
                    st.subheader(q.get('question', 'Question not found'))
                    selected_index = None
                    if st.session_state.student_answers and st.session_state.student_answers[i] in q.get('options', []):
                        selected_index = q['options'].index(st.session_state.student_answers[i])

                    answer = st.radio(
                        "Choose one option",
                        q.get('options', []),
                        index=selected_index,
                        key=f"{q.get('question', f'question_{i}')}_{i}"
                    )
                    if st.session_state.student_answers:
                        st.session_state.student_answers[i] = answer
                else:
                    st.write(f"Invalid question format at index {i}: {q}")

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
                    st.session_state.evaluation_done = True
        else:
            if st.session_state.exam_started:
                st.write("Questions were not properly generated.")

    if not st.session_state.exam_started and not st.session_state.evaluation_done:
        st.write(f"Current keyword: {keywords}, Experience level: {experience}")
        st.write("Click 'Start Exam' to begin.")


if task == "Code Evaluation":
    # Streamlit UI
    st.title("Welcome to Programming Test")

    # Default values
    keywords = "Python"
    experience = "3"

    # Get query parameters from URL
    query_params = st.query_params
    if "keywords" in query_params:
        keywords = query_params["keywords"]
    if "experience" in query_params:
        experience = query_params["experience"]

    # Initialize session state
    if 'questions' not in st.session_state:
        st.session_state.questions = None
        st.session_state.student_codes = {}
        st.session_state.evaluations = {}
        st.session_state.current_question_index = 0
        st.session_state.exam_started = False
        st.session_state.evaluation_done = False

    # Start Exam Button
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

    # Exam Logic
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
                if evaluation:
                    st.session_state.evaluations[st.session_state.current_question_index] = evaluation

                # Move to next question or finish the exam
                if st.session_state.current_question_index + 1 < len(st.session_state.questions):
                    st.session_state.current_question_index += 1
                    st.rerun()
                else:
                    # All questions completed, set evaluation_done to True
                    st.session_state.evaluation_done = True

    # Display Evaluations
    if st.session_state.evaluation_done:
        st.subheader("Evaluations:")
        for i, question in enumerate(st.session_state.questions):
            st.subheader(f"Evaluation for Question {i + 1}:")
            evaluation = st.session_state.evaluations.get(i)
            if evaluation and isinstance(evaluation, dict) and 'score' in evaluation and 'feedback' in evaluation:
                st.write(f"Score: {evaluation['score']}%")
                st.write(f"Feedback: {evaluation['feedback']}")
            else:
                st.write("Evaluation not available or invalid format.")

    # Default UI
    if not st.session_state.exam_started and not st.session_state.evaluation_done:
        st.write(f"Current keyword: {keywords}, Experience level: {experience}")
        st.write("Click 'Start Exam' to begin.")

if task == "Answer Evaluation":
    # Streamlit UI
    st.title("Welcome to Subjective Programming Test")

    # Default values
    keywords = "Python"
    experience = "3 years"

    # Get query parameters from URL
    query_params = st.query_params
    if "keywords" in query_params:
        keywords = query_params["keywords"]
    if "experience" in query_params:
        experience = query_params["experience"]

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
