# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from cli.output import CLIOutput
from cli.user_input import CLIUserInput


class CLIDay():

    # constants
    INTRO_TEXT_WIDTH = 60

    CMD_HELP_ALIASES   = [ "h", "help"  ]
    CMD_WORDS_ALIASES  = [ "w", "words" ]
    CMD_SKIP_ALIASES   = [ "s", "skip"  ]
    CMD_EXIT_ALIASES   = [ "e", "exit",
                           "q", "quit"  ]
    CMD_NEXT_ALIASES   = [ "n", "next"  ]
    CMD_PREV_ALIASES   = [ "p", "prev"  ]

    ACTION_EXIT             = "exit"
    ACTION_TITLE            = "title"
    ACTION_NEW_WORDS        = "new words"
    ACTION_INTRO_TEXT       = "intro text"
    ACTION_SAMPLE_SENTENCES = "sample sentences"
    ACTION_DEFINITIONS      = "definitions"
    ACTION_MATCHING         = "matching"
    ACTION_OTHER_NEW_WORDS  = "other new words"

    # General variables #
    _next_action = None
    _day = None

    @classmethod
    def start(cls, day):
        cls._day = day
        cls.mainloop()

    @classmethod
    def mainloop(cls):
    
        cls._next_action = "title"

        while cls._next_action != cls.ACTION_EXIT:

            if cls._next_action == cls.ACTION_TITLE:
                cls._next_action = cls.ACTION_NEW_WORDS
                cls.title()

            elif cls._next_action == cls.ACTION_NEW_WORDS:
                cls._next_action = cls.ACTION_INTRO_TEXT
                cls.new_words()
                CLIOutput.empty_line(1)
                CLIUserInput.wait_for_enter()

            elif cls._next_action == cls.ACTION_INTRO_TEXT:
                cls._next_action = cls.ACTION_SAMPLE_SENTENCES
                cls.intro_text()
                CLIOutput.empty_line(1)
                CLIUserInput.wait_for_enter()

            elif cls._next_action == cls.ACTION_SAMPLE_SENTENCES:
                cls._next_action = cls.ACTION_DEFINITIONS
                cls.sample_sentences()

            elif cls._next_action == cls.ACTION_DEFINITIONS:
                cls._next_action = cls.ACTION_MATCHING
                cls.definitions()

            elif cls._next_action == cls.ACTION_MATCHING:
                cls._next_action = cls.ACTION_OTHER_NEW_WORDS
                cls.matching()

            elif cls._next_action == cls.ACTION_OTHER_NEW_WORDS:
                cls._next_action = cls.ACTION_EXIT
                cls.other_new_words()

            else:
                raise KeyError("Unknown action request.")

# day displays ------------------------------------------------------- #

    @classmethod
    def title(cls):
        """Display title"""

        CLIOutput.empty_line(1)
        CLIOutput.center(cls._day.get_title())

    @classmethod
    def new_words(cls, display_in_full=True):
        """Display new words section"""
        
        regular = list()
        phonetic = list()

        for unit in cls._day.get_new_words():
            regular.append(unit["regular"])
            phonetic.append(unit["phonetic"])

        if display_in_full:
            CLIOutput.section_title("NEW WORDS")
            CLIOutput.empty_line(1)

        CLIOutput.empty_line(1)
        CLIOutput.words_table(regular, phonetic)

    @classmethod
    def intro_text(cls):
        """Display intro text"""

        parts = cls._day.get_intro_text()

        CLIOutput.empty_line(2)
        CLIOutput.framed(parts, cls.INTRO_TEXT_WIDTH)

# task answer cycle -------------------------------------------------- #

    @classmethod
    def _answer_cycle(cls, prompt, l_pr_question, answers, l_pr_answer, prev_action, l_prev_msg, l_next_msg):
        """Answer cycle"""

        while True:
            CLIOutput.empty_line(1)
            a_type, a_content = CLIUserInput.get_answer(prompt)

            if a_type == CLIUserInput.TYPE_ANSWER:

                if a_content in answers:
                    CLIOutput.empty_line(1)
                    l_pr_answer()
                    CLIOutput.empty_line(1)
                    CLIOutput.simple("Correct!")

                    return True
                else:
                    CLIOutput.warning("Incorrect, try again.")

            elif a_type == CLIUserInput.TYPE_COMMAND:
                if a_content in cls.CMD_WORDS_ALIASES:
                    cls.new_words(False)
                    CLIOutput.empty_line(1)
                    l_pr_question()
                elif a_content in cls.CMD_SKIP_ALIASES:
                    return True
                elif a_content in cls.CMD_NEXT_ALIASES:
                    l_next_msg()
                    return False
                elif a_content in cls.CMD_PREV_ALIASES:
                    l_prev_msg()
                    cls._next_action = prev_action
                    return False
                elif a_content in cls.CMD_EXIT_ALIASES:
                    cls._next_action = cls.ACTION_EXIT
                    return False
                elif a_content in cls.CMD_HELP_ALIASES:
                    cls.help_cmd_in_task()
                else:
                    CLIOutput.warning("Invalid command.")

            else:
                raise ValueError("Unknown answer type.")

# tasks -------------------------------------------------------------- #

    @classmethod
    def sample_sentences(cls):
        """Display 'sample sentences' task"""

        data = cls._day.get_sample_sentences()

        CLIOutput.section_title("SAMPLE SENTENCES")
        
        CLIOutput.empty_line(1)
        CLIOutput.simple(data["prompt"])

        CLIOutput.empty_line(1)

        for sentence in data["sentences"]:
            CLIOutput.numbered_sentence(sentence["id"], sentence["beginning"] + CLIOutput.BLANK + sentence["end"], CLIOutput.FORMAT_INDENTED)

        new_words_extension = cls._day.get_new_words_extension()

        CLIOutput.new_words_extension(new_words_extension)

        CLIOutput.empty_line(1)

        for sentence in data["sentences"]:

            prompt = "{}. ".format(sentence["id"])

            l_pr_question = lambda : CLIOutput.numbered_sentence(sentence["id"], sentence["beginning"] + CLIOutput.BLANK + sentence["end"], CLIOutput.FORMAT_REGULAR)

            answers = list()
            answers.append(sentence['answer'])

            full_answer = sentence['answer']
            if len(sentence["beginning"]) > 0:
                full_answer = sentence["beginning"] + " " + full_answer
            if len(sentence["end"]) > 0:
                if sentence["end"] not in [".", "!", "?", "?!", "!?"]:
                    full_answer += " "
                full_answer += sentence["end"]
            l_pr_answer = lambda : CLIOutput.simple(full_answer)

            prev_action = cls.ACTION_SAMPLE_SENTENCES

            l_prev_msg = lambda : CLIOutput.general_message("This is the first task: Starting from the beginning.")

            l_next_msg = lambda : None

            # answer cycle

            cls.new_words(False)
            CLIOutput.empty_line(1)
            l_pr_question()

            if not cls._answer_cycle(prompt, l_pr_question, answers, l_pr_answer, prev_action, l_prev_msg, l_next_msg):
                return

            # return after answer cycle returns


    @classmethod
    def definitions(cls):
        """Display 'definitions' task"""

        # skip until data files are complete
        return

        data = cls._day.get_definitions()

        CLIOutput.section_title("DEFINITIONS")

        CLIOutput.empty_line(1)
        CLIOutput.simple(data["prompt"])

        CLIOutput.empty_line(1)
        for definition in data["definitions"]:
            CLIOutput.numbered_sentence(definition["id"], definition["text"], CLIOutput.FORMAT_INDENTED)

        l_words = lambda : [CLIOutput.numbered_sentence(word["id"], word["text"], CLIOutput.FORMAT_INDENTED) for word in data["words"]]

        for definition in data["definitions"]:

            prompt = "{}. ".format(definition["id"])

            l_pr_question = lambda : CLIOutput.numbered_sentence(definition["id"], definition["text"], CLIOutput.FORMAT_REGULAR)

            answers = list()
            answer_id = [value  for (id, value) in data["answers"] if id == definition["id"]][0]
            answers.append(answer_id)
            answer_text = [item["text"]  for item in data["words"] if item["id"] == answer_id][0]
            answers.append(answer_text)

            l_pr_answer = lambda : CLIOutput.numbered_sentence(answer_id, answer_text, CLIOutput.FORMAT_REGULAR)

            prev_action = cls.ACTION_SAMPLE_SENTENCES

            l_prev_msg = lambda : None

            l_next_msg = lambda : None

            # answer cycle

            CLIOutput.empty_line(2)
            l_words()
            CLIOutput.empty_line(1)
            l_pr_question()

            if not cls._answer_cycle(prompt, l_pr_question, answers, l_pr_answer, prev_action, l_prev_msg, l_next_msg):
                return

            # return after answer cycle returns


    @classmethod
    def matching(cls):
        """Display 'matching' task"""

        # skip until data files are complete
        return

        data = cls._day.get_matching()

        CLIOutput.section_title(data["name"])

        CLIOutput.empty_line(1)
        CLIOutput.simple(data["prompt"])

        CLIOutput.empty_line(1)
        for sentence in data["sentences"]:
            CLIOutput.numbered_sentence(sentence["id"], sentence["text"], CLIOutput.FORMAT_INDENTED)

        l_words = lambda : [CLIOutput.numbered_sentence(word["id"], word["text"], CLIOutput.FORMAT_INDENTED) for word in data["words"]]

        for sentence in data["sentences"]:

            prompt = "{}. ".format(sentence["id"])

            l_pr_question = lambda : CLIOutput.numbered_sentence(sentence["id"], sentence["text"], CLIOutput.FORMAT_REGULAR)

            answers = list()
            answer_id = [value  for (id, value) in data["answers"] if id == sentence["id"]][0]
            answers.append(answer_id)
            answer_text = [item["text"]  for item in data["words"] if item["id"] == answer_id][0]
            answers.append(answer_text)

            l_pr_answer = lambda : CLIOutput.numbered_sentence(answer_id, answer_text, CLIOutput.FORMAT_REGULAR)

            prev_action = cls.ACTION_SAMPLE_SENTENCES

            l_prev_msg = lambda : None

            l_next_msg = lambda : None

            # answer cycle

            CLIOutput.empty_line(2)
            l_words()
            CLIOutput.empty_line(1)
            l_pr_question()

            if not cls._answer_cycle(prompt, l_pr_question, answers, l_pr_answer, prev_action, l_prev_msg, l_next_msg):
                return

            # return after answer cycle returns


    @classmethod
    def other_new_words(cls):
        """Display other new words section"""

        data = cls._day.get_other_new_words()

        CLIOutput.section_title("OTHER NEW WORDS:")

        CLIOutput.empty_line(1)
        CLIOutput.simple(data["prompt"])

        CLIOutput.empty_line(1)
        a_type, a_content = CLIUserInput.get_answer("")

        CLIOutput.empty_line(1)

# helper ------------------------------------------------------------- #

    @classmethod
    def help_cmd_in_task(cls):
    
        collection = [
            ["words", "Display New Words section again."],
            ["skip",  "Move on to the next part of the task."],
            ["next",  "Leave task and move on to the next one."],
            ["prev",  "Leave task and jump back to the previous one."],
            ["exit",  "Leave task an exit to top program level."]
        ]

        CLIOutput.empty_line(1)
        CLIOutput.simple("Within the task, the following commands are available:")
        CLIOutput.value_pair_list(collection, CLIOutput.FORMAT_REGULAR, CLIOutput.SPACING_CLOSE)

# END ---------------------------------------------------------------- #
