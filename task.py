import numpy as np


class Task:

    EVAL_POLICY_STOP_AT_FIRST_INCORRECT = "stop_at_first_incorrect"
    EVAL_POLICY_CONTINUE_UNTIL_END = "continue_until_end"
    """
    A class to represent a task.
    Question and answer_column can both be either a string to represent a column name in the dataset,
    or a python function. A function for question or answer_column takes in a row of the dataset and returns a string.
    A function for answer_column takes a row of the dataset and a string, and returns either a a list of numbers.
    """

    def __init__(
        self,
        tokenizer,
        name,
        dataset_name,
        prompt,
        question,
        answer_column,
        ignore_case=True,
        strip=True,
        treat_equal_numbers_same=True,
        custom_eval=None,
        eval_policy=EVAL_POLICY_STOP_AT_FIRST_INCORRECT,
    ):
        self.tokenizer = tokenizer
        self.name = name
        self.dataset_name = dataset_name
        self.prompt = prompt
        self.question = question
        self.answer_column = answer_column
        self.ignore_case = ignore_case
        self.strip = strip
        self.treat_equal_numbers_same = treat_equal_numbers_same
        self.custom_eval = custom_eval
        self.eval_policy = eval_policy

    def default_eval(self, row, given_answer):
        if callable(self.answer_column):
            correct_answer = self.answer_column(row, given_answer)
        else:
            correct_answer = row[self.answer_column]

        # test if is list. Then we assume it is a list of correct answers.
        # use the one with most reward
        if isinstance(correct_answer, list):
            rewards = [self.compare(given_answer, ca) for ca in correct_answer]
            return max(rewards, key=lambda x: sum(x))
        else:
            return self.compare(given_answer, correct_answer)

    """
    returns a list of numbers, where each number is the reward for the corresponding token in the given answer.
    """

    def compare(self, given_answer, correct_answer):
        # test if correct_answer is a number, transform it to string
        if isinstance(correct_answer, (int, float)):
            correct_answer = str(correct_answer)

        if self.ignore_case:
            correct_answer = correct_answer.lower()
            given_answer = given_answer.lower()

        if self.trim:
            correct_answer = correct_answer.strip()
            given_answer = given_answer.strip()

        # test if given_answer is string form of a number
        if self.treat_equal_numbers_same:
            try:
                given_answer_float = float(given_answer)
                correct_answer_float = float(correct_answer)
                if given_answer_float == correct_answer_float:
                    correct_answer = given_answer
            except ValueError:
                pass

        tok_correct_answer = self.tokenizer.tokenize(
            correct_answer, add_special_tokens=False
        )
        tok_given_answer = self.tokenizer.tokenize(
            given_answer, add_special_tokens=False
        )

        result = np.array(
            [1 if a == b else 0 for a, b in zip(tok_given_answer, tok_correct_answer)]
        )
        if len(tok_given_answer) > len(tok_correct_answer):
            result = np.pad(
                result, (0, len(tok_given_answer) - len(tok_correct_answer))
            )
        result /= len(tok_correct_answer)
        if self.eval_policy == self.EVAL_POLICY_STOP_AT_FIRST_INCORRECT:
            # only give reward until the first incorrect token
            # fill after first incorrect token with 0
            result[np.argmax(result == 0) :] = 0

        return result

    def evaluate(self, row, given_answer):
        if self.custom_eval:
            return self.custom_eval(row, given_answer)
        else:
            return self.default_eval(row, given_answer)

    def get_question(self, row):
        if callable(self.question):
            return self.question(row)
        elif isinstance(self.question, str):
            return row[self.question]
        elif isinstance(self.question, list):
            return "\n\n".join([row[q] for q in self.question])
